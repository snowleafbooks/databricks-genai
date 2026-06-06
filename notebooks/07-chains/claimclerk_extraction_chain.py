# ClaimClerk extraction chain — standalone PyFunc module for code-based logging.
#
# This file is the artifact that `mlflow.pyfunc.log_model(python_model=...)`
# logs. The driver notebook (c0701-build-claimclerk-extraction-chain.py)
# imports it for in-process testing AND passes its filename to `log_model`
# so MLflow re-executes the file fresh at serve time (the code-based-logging
# pattern).
#
# Shape: PyFunc subclass wrapping a three-component chain (prompt template +
# LLM + JSON-validate). Pre-processing: ai_mask-style PII strip on the email
# body. Post-processing: JSON-schema validation + envelope wrap (timestamp +
# version_marker).
#
# Source: https://docs.databricks.com/aws/en/generative-ai/agent-framework/log-agent

import json
import re
import time

import mlflow
import pandas as pd

# NOTE: databricks-langchain and any other heavy deps are imported inside
# load_context, not at module scope. Model Serving installs pip_requirements
# AFTER loading the module file — top-level imports of those packages cause
# ModuleNotFoundError at deploy time.

THROTTLE_SECONDS = 0.3
MAX_RETRIES = 4

# The prompt is the Registry artefact registered by the c0301 notebook
# as `genaicert.pawshield.claimclerk_extraction@champion`. The driver
# notebook (c0701) resolves it at build time and bakes the literal template
# into `model_config['prompt_template']`; this module's load_context uses
# that baked template directly and only calls
# `mlflow.genai.load_prompt(model_config['prompt_uri'])` as a fallback when
# no template was baked. Baking at build time means the served version is
# byte-exact replayable from the registered model alone — no live Registry
# lookup at serve time. The PolicyPal chain (policypal_chain.py) is shipped
# the same way (template-first, load_prompt as fallback), so promotion for
# both is a re-log + redeploy, not an alias move alone.
#
# `DEFAULT_PROMPT_TEMPLATE` is the inline fallback used when the Prompt
# Registry hasn't been seeded yet (a brand-new workspace running this
# notebook before c0301 has registered the prompt). It is a SIMPLIFIED copy
# of the champion prompt — it omits the registered prompt's few-shot
# examples — kept only so the chain runs before c0301 has registered it.
DEFAULT_PROMPT_TEMPLATE = (
    "You are an email-parsing assistant for a pet insurance claims team.\n\n"
    "TASK\nExtract structured information from a customer email.\n\n"
    "CONSTRAINTS\n"
    "Output a single JSON object with exactly these keys:\n"
    "- claim_id        (string, format CLM-YYYY-MM-NNNNN, or null)\n"
    "- customer_id     (string, format CUST-NAME-NNN, or null)\n"
    "- pet_name        (string, or null)\n"
    "- contact_phone   (string in hyphenated format like 512-555-0188, or null)\n"
    "- intent          (one of: claim_status, policy_q, appeal, complaint,\n"
    "                          vet_lookup, emergency, other)\n"
    "- sentiment       (one of: positive, neutral, frustrated, angry, panicked)\n"
    "- urgency         (one of: low, medium, high, emergency)\n\n"
    "Do not include any text before or after the JSON. Do not add fields.\n"
    "Use null for fields the email does not mention; for intent / sentiment /\n"
    "urgency always emit one of the listed enum values (use 'other' / "
    "'neutral' / 'low' when unclear)."
)

# Output JSON schema — used by the postprocess validator. Drift here would
# silently corrupt downstream consumers (the batch extract, the eval set,
# and the agent reading customer_id / claim_id). The schema is enforced as the
# last gate before the chain returns; an off-schema LLM emission raises a
# typed error the served endpoint surfaces as a 422.
OUTPUT_SCHEMA = {
    "type": "object",
    "required": [
        "claim_id", "customer_id", "pet_name", "contact_phone",
        "intent", "sentiment", "urgency",
    ],
    "properties": {
        "claim_id":      {"type": ["string", "null"]},
        "customer_id":   {"type": ["string", "null"]},
        "pet_name":      {"type": ["string", "null"]},
        "contact_phone": {"type": ["string", "null"]},
        # intent/sentiment/urgency: enum-or-null. The champion prompt instructs
        # always-emit-a-string + has the "other" / "neutral" / "low" enum
        # values to express "no signal here", but Llama 3.1 8B
        # occasionally interprets the prompt's general "Use null for any
        # field the email does not mention" line as applying here too
        # and emits null. The contract still rejects garbage strings via
        # the enum; the null-axis is loosened to keep the chain robust
        # to small-model variance. The `response_format=json_schema`
        # path is the doc-preferred way to make the LLM emit
        # schema-conforming output by construction; the loosened-null
        # contract here is the trade-off when that path is not used.
        "intent": {
            "oneOf": [
                {"type": "string", "enum": [
                    "claim_status", "policy_q", "appeal", "complaint",
                    "vet_lookup", "emergency", "other",
                ]},
                {"type": "null"},
            ],
        },
        "sentiment": {
            "oneOf": [
                {"type": "string", "enum": [
                    "positive", "neutral", "frustrated", "angry", "panicked",
                ]},
                {"type": "null"},
            ],
        },
        "urgency": {
            "oneOf": [
                {"type": "string", "enum": [
                    "low", "medium", "high", "emergency",
                ]},
                {"type": "null"},
            ],
        },
    },
    "additionalProperties": False,
}

# Phone-number redaction pattern — the source-side PII strip that is the canonical pre-LLM redaction surface. The serving-side
# strip here is the second layer: redact phone numbers from the email body
# BEFORE the LLM ever sees them. The schema still asks the LLM to emit
# contact_phone — but the LLM only sees the redaction token, so it emits
# null. Downstream consumers needing the real phone read it from the
# customer record, not from the chain's output.
PHONE_RE = re.compile(
    r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b"
)


class ClaimClerkExtraction(mlflow.pyfunc.PythonModel):
    """PyFunc wrapping the ClaimClerk extraction chain (prompt + LLM +
    JSON-validate). Inputs: a DataFrame with one `email_body` column (raw
    email text). Outputs: a list of validated extraction envelopes —
    `{extracted: <JSON>, version_marker, request_timestamp}`."""

    def load_context(self, context):
        config = context.model_config or {}
        self._llm_endpoint = config.get(
            "llm_endpoint", "databricks-meta-llama-3-1-8b-instruct"
        )
        # The prompt template is resolved at BUILD time (in the driver
        # notebook before log_model), then injected here via model_config
        # so the load is offline at serve time. Falls back to the inline
        # template if neither prompt_template nor prompt_uri is supplied.
        if "prompt_template" in config:
            self._prompt_template = config["prompt_template"]
            self._prompt_uri = config.get("prompt_uri", "inline")
        elif "prompt_uri" in config:
            self._prompt_uri = config["prompt_uri"]
            self._prompt_template = mlflow.genai.load_prompt(
                self._prompt_uri
            ).template
        else:
            self._prompt_template = DEFAULT_PROMPT_TEMPLATE
            self._prompt_uri = "inline-default"
        self._version_marker = config.get("version_marker", "unknown")
        from databricks_langchain import ChatDatabricks
        # `extra_params={"response_format": {"type": "json_object"}}`
        # constrains the LLM to emit valid JSON by construction
        # (Databricks-served Llama 3.x supports this). Without it,
        # Llama 3.1 8B will sometimes ignore the prompt and emit
        # JavaScript parser code or markdown prose around the JSON —
        # the brace-balanced extractor in `_validate` is the second-
        # layer fallback for the small-model edge case.
        self._llm = ChatDatabricks(
            endpoint=self._llm_endpoint,
            temperature=0.0,
            extra_params={"response_format": {"type": "json_object"}},
        )

    def _strip_pii(self, email_body: str) -> str:
        return PHONE_RE.sub("[PHONE_REDACTED]", email_body)

    def _validate(self, raw: str) -> dict:
        """Postprocess: parse + schema-validate the LLM's JSON emission.

        Llama 3.1 8B (and similar small models) often wrap JSON in prose
        ("Here's the extracted information:") or markdown fences despite
        the prompt's no-prose instruction. The two-stage extractor below
        first strips known wrappers, then falls back to a brace-balanced
        substring scan to pull the first valid JSON object out of any
        surrounding text. The schema check after is the contract gate;
        the extractor is only there to make the chain robust to small-
        model emission variance.
        """
        text = (raw or "").strip()
        if not text:
            raise ValueError("LLM returned empty output (no JSON to validate)")
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
        # Brace-balanced extraction: walk the string, track depth, return
        # the substring from the first `{` to the depth-zero `}` that
        # closes it. Handles "prose then JSON" and "JSON then prose"
        # without false-matching `}` characters inside strings naively
        # (json.loads will reject malformed slices, so we fall through).
        candidate = text
        if not candidate.startswith("{"):
            start = candidate.find("{")
            if start >= 0:
                depth = 0
                end = -1
                for i, ch in enumerate(candidate[start:], start=start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end > start:
                    candidate = candidate[start:end]
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM emitted non-JSON output: {e}; raw start: {text[:120]!r}"
            ) from e
        from jsonschema import validate, ValidationError
        try:
            validate(instance=obj, schema=OUTPUT_SCHEMA)
        except ValidationError as e:
            raise ValueError(f"LLM output failed schema: {e.message}") from e
        return obj

    def _safe_invoke(self, messages: list) -> str:
        """Rate-limit guard: throttle + retry on 429.

        Takes a LangChain `messages` list (SystemMessage + HumanMessage)
        so the registered prompt lands as the system role and the email
        lands as the user role — matching c0301's `chat.completions`
        shape. Treating the registered template as a single combined
        prompt fed to `.invoke(str)` was the bug that made Llama 3.1 8B
        emit JavaScript parser code instead of extracted JSON.
        """
        for attempt in range(MAX_RETRIES):
            try:
                out = self._llm.invoke(messages)
                time.sleep(THROTTLE_SECONDS)
                return out.content
            except Exception as e:
                msg = str(e)
                is_429 = "REQUEST_LIMIT_EXCEEDED" in msg or "429" in msg
                if not is_429 or attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** (attempt + 1))

    def _run_one(self, email_body: str) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        stripped = self._strip_pii(email_body)
        # System message = registered extraction instruction (loaded from
        # Prompt Registry at build time, baked into the artifact).
        # User message = the PII-stripped email body. This split matches
        # the OpenAI-compatible chat shape c0301 uses for the SAME
        # registered prompt.
        #
        # The user message is prefixed with "Extract as json:" because
        # when `response_format={"type": "json_object"}` is set, the
        # Databricks OpenAI-compatible API requires at least one message
        # to contain the lowercase substring "json". The system prompt
        # may use "JSON object" (uppercase), which the case-sensitive
        # check skips, so the user-side prefix guarantees the substring
        # regardless of how the baked template is cased.
        messages = [
            SystemMessage(content=self._prompt_template),
            HumanMessage(content=f"Extract as json:\n\n{stripped}"),
        ]
        raw = self._safe_invoke(messages)
        extracted = self._validate(raw)
        return {
            "extracted": json.dumps(extracted),
            "version_marker": self._version_marker,
            "prompt_uri": self._prompt_uri,
            "request_timestamp_unix": time.time(),
        }

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, dict):
            model_input = pd.DataFrame([model_input])
        return [
            self._run_one(row["email_body"])
            for _, row in model_input.iterrows()
        ]


# Code-based-logging entrypoint: MLflow re-executes this file at serve time
# and uses the instance registered here as the served PyFunc.
mlflow.models.set_model(ClaimClerkExtraction())
