# PolicyClassifierAgent — standalone ResponsesAgent module for
# code-based logging.
#
# This file is the artifact that `mlflow.pyfunc.log_model(python_model=...)`
# logs. The Databricks driver notebook (c1501-mlflow-genai-end-to-end.py)
# imports it for in-process testing (eval + optimize phases) AND passes
# its filename to `log_model` so MLflow re-executes the file fresh at
# serve time. Workspace notebooks uploaded with `--format SOURCE` have
# no `.py` on disk, so `python_model` must point at a real .py workspace
# file in the same folder — that's this file.
#
# Source: https://docs.databricks.com/aws/en/generative-ai/agent-framework/log-agent

import time
import uuid
from typing import Literal

import mlflow
from databricks_openai import DatabricksOpenAI
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from openai import RateLimitError

THROTTLE_SECONDS = 0.3
MAX_RETRIES = 4

INTENT_LABELS = ("claim", "question", "appeal", "complaint")
IntentLabel = Literal["claim", "question", "appeal", "complaint"]

# Inline fallback the agent uses if the Prompt Registry lookup hasn't been
# seeded yet (driver notebook's Phase 2 registers the prompt). Production
# reads from the registry only.
DEFAULT_PROMPT_TEMPLATE = (
    "You are PolicyPal's intent router for PawShield. "
    "Read the customer email and return EXACTLY one word from this set: "
    "{labels}. "
    "Output the single label only — no punctuation, no explanation."
)

DEFAULT_LLM_ENDPOINT = "databricks-meta-llama-3-1-8b-instruct"


class PolicyClassifierAgent(ResponsesAgent):
    """Tiny classifier agent. Resolves its system prompt from the MLflow
    Prompt Registry in `load_context` — once per replica warm-up — so an
    alias move takes effect on the next warm-up without re-logging the
    model file (the documented, network-cheap placement for runtime alias
    resolution). Configuration (prompt URI + LLM endpoint) is read from
    `model_config` at `load_context` time so one class serves multiple
    deployments via different log_model invocations."""

    def load_context(self, context):
        config = (context.model_config if context else None) or {}
        self._prompt_uri = config.get("prompt_uri", "")
        self._llm_endpoint = config.get("llm_endpoint", DEFAULT_LLM_ENDPOINT)
        # Source: https://api-docs.databricks.com/python/databricks-ai-bridge/latest/databricks_openai.html
        self._client = DatabricksOpenAI()
        # Resolve the prompt ONCE here (load_context is the once-per-replica
        # hook) and cache it, rather than calling load_prompt on every
        # request. An alias move is picked up on the next replica warm-up
        # (or after update_endpoint forces a recycle), without re-logging.
        try:
            tmpl = mlflow.genai.load_prompt(self._prompt_uri).template
        except Exception:
            tmpl = DEFAULT_PROMPT_TEMPLATE
        self._system_prompt = tmpl.format(labels=", ".join(INTENT_LABELS))

    @mlflow.trace(span_type="LLM")
    def _classify(self, *, email_text: str) -> IntentLabel:
        """Classify a single email into one of the four intent labels.

        Args:
            email_text: The raw customer email body.

        Returns:
            One of ``claim``, ``question``, ``appeal``, ``complaint``.
            If the LLM returns something off-vocabulary, falls back to
            ``question`` (the safest no-action label).
        """
        # Prompt was resolved once in load_context (per-replica), not per
        # call — an alias move lands on the next warm-up.
        system_prompt = self._system_prompt

        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=self._llm_endpoint,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": email_text},
                    ],
                    temperature=0.0,
                    max_tokens=8,
                )
                time.sleep(THROTTLE_SECONDS)
                break
            except RateLimitError:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** (attempt + 1))
        raw = (resp.choices[0].message.content or "").strip().lower()
        token = raw.split()[0].strip(".,!?:;\"'") if raw else "question"
        return token if token in INTENT_LABELS else "question"

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        email_text = ""
        for msg in request.input:
            if getattr(msg, "role", None) == "user":
                email_text = getattr(msg, "content", "") or ""
        label = self._classify(email_text=email_text)
        return ResponsesAgentResponse(output=[{
            "type": "message",
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": [{"type": "output_text", "text": label}],
        }])


# Code-based-logging entrypoint: MLflow re-executes this file at serve
# time and uses the instance registered here. The instance is bare —
# `load_context` reads model_config and constructs clients.
mlflow.models.set_model(PolicyClassifierAgent())
