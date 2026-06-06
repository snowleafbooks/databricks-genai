# ClaimClerk extraction chain — bare LangChain flavor (no pre/post-processing).
#
# This file is the artifact that `mlflow.langchain.log_model(lc_model=...)`
# logs. LangChain v1+ requires the code-based-logging shape:
# `lc_model` is a PATH to a .py file that constructs the chain and
# registers it via `mlflow.models.set_model(chain)`. The live-object
# form (`lc_model=chain_instance`) is deprecated.
#
# Pedagogical point: the langchain flavor is no longer a "leaner direct
# path" relative to pyfunc — both flavors now use the same code-based-
# logging shape. The remaining difference is which lineage MLflow
# captures (LangChain-specific chain spec for langchain.log_model;
# generic PyFunc signature for pyfunc.log_model).
#
# Source: https://mlflow.org/docs/latest/ml/model/models-from-code/

import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

# Inline prompt — a simplified copy of the champion prompt (it omits
# the registered prompt's few-shot examples). The PyFunc chain loads from
# the Prompt Registry at build time; this demo file inlines for simplicity
# (the file's purpose is to exercise the langchain.log_model surface, not
# to model the full registry-integration shape). The Registry artefact
# `claimclerk_extraction@champion` is the source of truth for the prompt.
_SYSTEM_PROMPT = (
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
    "urgency always emit one of the listed enum values."
)

_llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-1-8b-instruct",
    temperature=0.0,
    extra_params={"response_format": {"type": "json_object"}},
)


def _build_messages(payload: dict) -> list:
    # Payload is `{"email_body": "..."}`. The "Extract as json:" prefix
    # satisfies the Databricks-side `response_format=json_object`
    # enforcement that messages must contain lowercase "json".
    return [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Extract as json:\n\n{payload['email_body']}"),
    ]


bare_chain = (
    RunnableLambda(_build_messages)
    | _llm
    | RunnableLambda(lambda msg: msg.content)
)

# Code-based-logging entrypoint. MLflow re-executes this file at serve
# time and uses the runnable registered here as the served chain.
mlflow.models.set_model(bare_chain)
