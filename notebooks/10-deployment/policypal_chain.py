# PolicyPal chain — standalone PyFunc module for code-based logging.
#
# This file is the artifact that `mlflow.pyfunc.log_model(python_model=...)`
# logs. The Databricks driver notebook (c1001-deploy-policypal.py) imports
# it for in-process testing AND passes its filename to `log_model` so MLflow
# re-executes the file fresh at serve time (the code-based-logging pattern).
#
# Source: https://docs.databricks.com/aws/en/generative-ai/agent-framework/log-agent

import time

import mlflow
import pandas as pd
# NOTE: databricks-vectorsearch and databricks-langchain are NOT imported
# at module scope. Model Serving installs pip_requirements AFTER loading
# the module file — top-level imports of those packages cause
# "No module named 'databricks.vector_search'" at deploy time.
# Import them inside load_context() instead, which runs after deps install.

THROTTLE_SECONDS = 0.3
MAX_RETRIES = 4

# The chain's prompt is IN-PLACE TEMPLATE ONLY. `model_config` carries the
# literal prompt text in `prompt_template` (baked into the logged model at
# log_model time); `load_context` reads it directly and never calls the
# Prompt Registry at serve time — a served Custom (PyFunc) endpoint's
# auto-auth SP cannot read the Registry, so any serve-time load_prompt makes
# the endpoint fail to come READY (UPDATE_FAILED). The driver (c1001 §2.5)
# still registers v1 + v2 in the Registry for versioning/lineage and sets
# @champion / @candidate, but the served chain consumes the BAKED template,
# not the Registry. A/B testing is two log_model calls baking a DIFFERENT
# prompt_template into each — so promotion is a re-log + redeploy, not an
# alias move alone.
#
# DEFAULT_PROMPT_TEMPLATE is the launch prompt the c1001 driver registers
# as policypal_qa@champion (the driver imports this same constant, so they
# can't drift) — kept here so the source of
# truth for the launch prompt lives next to the chain code that consumes
# it, and so the driver can `from policypal_chain import DEFAULT_PROMPT_TEMPLATE`
# without re-typing the template.
DEFAULT_PROMPT_TEMPLATE = (
    "You are PolicyPal, PawShield's customer-help assistant.\n\n"
    "Answer the customer's question using ONLY the policy excerpts "
    "below. Cite the Section number when you quote a clause. If the "
    "excerpts don't answer the question, say so plainly — don't "
    "guess.\n\n"
    "Policy excerpts:\n{context}\n\n"
    "Customer question: {question}\n\n"
    "Answer:"
)


class PolicyPalChain(mlflow.pyfunc.PythonModel):
    """PyFunc wrapping the PolicyPal-shape chain (VS retrieval + prompt +
    LLM). The chain body runs the same retrieval/prompt/LLM logic as
    the inline `policypal_chain` form (behaviourally equivalent); the only
    addition is the PyFunc envelope so the chain is serveable and the
    model_config indirection that lets one class serve multiple prompt
    versions on one endpoint via `traffic_config`. The prompt is in-place
    template only: load_context reads `model_config['prompt_template']`
    (baked at log time) and never calls the Prompt Registry at serve time —
    the served SP can't read it. Promotion is therefore a re-log + redeploy,
    not an alias move alone."""

    def load_context(self, context):
        config = context.model_config or {}
        self._vs_endpoint = config["vs_endpoint"]
        self._index_name = config["index_name"]
        self._llm_endpoint = config["llm_endpoint"]
        # Prompt resolution is IN-PLACE TEMPLATE ONLY. A served Custom
        # (PyFunc) endpoint's auto-auth service principal cannot read the
        # MLflow Prompt Registry — a registered prompt is not a grantable
        # auto-auth resource, and MLflow 3 also auto-associates any prompt
        # loaded during the log run with the model, so a chain that called
        # mlflow.genai.load_prompt() at serve time (or a driver that loaded
        # the prompt while logging) would make the served SP try to read the
        # Registry at load → the endpoint fails to come READY (UPDATE_FAILED).
        # The c1001 driver therefore resolves the prompt at log time and bakes
        # the literal text into model_config["prompt_template"]; the chain
        # reads that directly and never touches the Registry at serve time.
        # (The Registry still holds the canonical record + @champion alias for
        # lineage; promotion is a re-log + redeploy with a new baked template,
        # not an alias move alone.)
        self._prompt_template = config["prompt_template"]
        self._prompt_uri = config.get("prompt_uri", "inline")
        # Per-deployment marker the chain echoes back in every response.
        # The driver notebook passes different values per version (v1
        # vs v2 in §3 / §7) so a client looping over the endpoint can
        # see which served entity handled each request without waiting
        # on the inference-table lag.
        self._version_marker = config.get("version_marker", "unknown")
        from databricks.vector_search.client import VectorSearchClient
        from databricks_langchain import ChatDatabricks

        self._vsc = VectorSearchClient(disable_notice=True)
        self._index = self._vsc.get_index(
            endpoint_name=self._vs_endpoint,
            index_name=self._index_name,
        )
        self._llm = ChatDatabricks(endpoint=self._llm_endpoint)

    def _safe_invoke(self, prompt: str):
        """Rate-limit guard: throttle + retry on 429."""
        for attempt in range(MAX_RETRIES):
            try:
                out = self._llm.invoke(prompt)
                time.sleep(THROTTLE_SECONDS)
                return out
            except Exception as e:
                msg = str(e)
                is_429 = "REQUEST_LIMIT_EXCEEDED" in msg or "429" in msg
                if not is_429 or attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** (attempt + 1))

    @staticmethod
    def _strip_pii_before_response(text: str) -> str:
        """Redact phone numbers from the answer before returning. PolicyPal is
        a Custom Model Serving (PyFunc) endpoint, so AI Gateway PII guardrails
        do not apply — PII handling lives in-chain here, as a response
        postprocess (mirrors the in-chain pattern ClaimClerk uses on input)."""
        import re
        return re.sub(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", "[redacted-phone]", text)

    def _run_one(self, question: str, state: str, tier: str) -> dict:
        retrieved = self._index.similarity_search(
            query_text=question,
            columns=["chunk_id", "doc_id", "section", "chunk_text"],
            num_results=4,
            filters={"state": state, "tier": tier},
        )
        rows = retrieved["result"]["data_array"]
        cols = [c["name"] for c in retrieved["manifest"]["columns"]]
        chunks = [dict(zip(cols, r)) for r in rows]
        context = "\n\n---\n\n".join(
            f"[{c['section']}] {c['chunk_text']}" for c in chunks
        )
        prompt = self._prompt_template.format(context=context, question=question)
        response = self._safe_invoke(prompt)
        return {
            "answer": self._strip_pii_before_response(response.content),
            "retrieved_chunk_ids": [c["chunk_id"] for c in chunks],
            "retrieved_sections": [str(c["section"]) for c in chunks],
            "version_marker": self._version_marker,
        }

    def predict(self, context, model_input, params=None):
        # Model Serving sends a pandas DataFrame with question/state/tier columns.
        if isinstance(model_input, dict):
            model_input = pd.DataFrame([model_input])
        return [
            self._run_one(row["question"], row["state"], row["tier"])
            for _, row in model_input.iterrows()
        ]


# Code-based-logging entrypoint: MLflow re-executes this file at serve time
# and uses the instance registered here as the served PyFunc.
mlflow.models.set_model(PolicyPalChain())
