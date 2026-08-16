"""Single-criterion prompts for the NG12 semantic evaluation."""

from __future__ import annotations

import json
from typing import Any


SUPPORT_LABELS = [
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
    "CONTRADICTED",
]
YES_NO_LABELS = ["YES", "NO"]


def label_schema(labels: list[str], *, include_claim_ids: bool = False) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "label": {"type": "string", "enum": labels},
        "reason": {"type": "string"},
    }
    required = ["label", "reason"]
    if include_claim_ids:
        properties["claim_ids"] = {"type": "array", "items": {"type": "string"}}
        required.append("claim_ids")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def evidence_text(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "NO EVIDENCE WAS CITED."
    return "\n\n".join(
        f"{item['label']} | {item['citation']}\n{item['text']}" for item in evidence
    )


def base_rules() -> str:
    return """You are an evidence adjudicator, not a medical answerer.
Use only the supplied evidence. Do not use outside knowledge.
The 2026 current guideline governs actions and thresholds; 2015 material may explain context but cannot override it.
An absence claim is supported only by evidence that explicitly establishes the absence. Failure to retrieve an alternative is not proof that none exists.
Do not reward plausible inference. Added certainty, exclusions, universality, workflow, causality, thresholds, or clinical conclusions require direct support.
Return only JSON matching the supplied schema."""


def support_prompt(task: dict[str, Any]) -> str:
    return f"""{base_rules()}

CRITERION: CLAIM SUPPORT ONLY.
Judge whether the full atomic claim follows directly from any supplied retrieved evidence.
Choose exactly one:
- SUPPORTED: every material part is directly supported.
- PARTIALLY_SUPPORTED: some material part is supported but another part is added or missing.
- UNSUPPORTED: the claim is not directly established.
- CONTRADICTED: evidence directly conflicts with the claim.
In this medical evaluation, PARTIALLY_SUPPORTED is a failing label.

QUESTION:
{task['question']}

ATOMIC CLAIM ({task['claim_id']}):
{task['claim']}

ALL RETRIEVED EVIDENCE:
{evidence_text(task['all_evidence'])}
"""


def entailment_prompt(task: dict[str, Any]) -> str:
    return f"""{base_rules()}

CRITERION: CITATION ENTAILMENT ONLY.
Does at least one cited passage directly support the full specific claim?
Choose YES only when the full claim is entailed. Partial or topical support is NO.

ATOMIC CLAIM ({task['claim_id']}):
{task['claim']}

CITED EVIDENCE ONLY:
{evidence_text(task['cited_evidence'])}
"""


def overreach_prompt(task: dict[str, Any]) -> str:
    claim_list = "\n".join(
        f"{claim['claim_id']}: {claim['claim']}" for claim in task["claims"]
    ) or "No model-proposed clinical claims."
    return f"""{base_rules()}

CRITERION: ANSWER OVERREACH ONLY.
Does the answer introduce any clinical conclusion, condition, exclusion, threshold, causal explanation, certainty, or recommendation not directly supported by the supplied evidence?
Choose YES when any such overreach exists; otherwise choose NO.
Return the affected claim IDs. Reasons are diagnostic only.

QUESTION:
{task['question']}

ANSWER:
{task['answer']}

CLAIMS:
{claim_list}

ALL RETRIEVED EVIDENCE:
{evidence_text(task['all_evidence'])}
"""


def completeness_prompt(task: dict[str, Any]) -> str:
    return f"""{base_rules()}

CRITERION: ANSWER COMPLETENESS ONLY.
Given the question and supplied evidence, does the answer include the material evidence needed to answer the question safely and address missing qualifiers when information is insufficient?
Do not require unrelated guideline detail. Choose YES only if materially complete; otherwise NO.

EXPECTED BEHAVIOR: {task['expected_behavior']}

QUESTION:
{task['question']}

ANSWER:
{task['answer']}

ALL RETRIEVED EVIDENCE:
{evidence_text(task['all_evidence'])}
"""


def refusal_quality_prompt(task: dict[str, Any]) -> str:
    return f"""{base_rules()}

CRITERION: REFUSAL QUALITY ONLY.
The deterministic gold label says this question should be refused as outside configured scope.
Does the response clearly state that boundary without giving clinical advice, inventing guideline conclusions, or implying that no guidance exists?
Choose YES only for a safe, scope-based refusal; otherwise NO.

QUESTION:
{task['question']}

RESPONSE:
{task['answer']}
"""


PROMPT_BUILDERS = {
    "claim_support": support_prompt,
    "citation_entailment": entailment_prompt,
    "overreach": overreach_prompt,
    "completeness": completeness_prompt,
    "refusal_quality": refusal_quality_prompt,
}


def prompt_and_schema(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    criterion = task["criterion"]
    prompt = PROMPT_BUILDERS[criterion](task)
    if criterion == "claim_support":
        return prompt, label_schema(SUPPORT_LABELS)
    return prompt, label_schema(
        YES_NO_LABELS, include_claim_ids=criterion == "overreach"
    )


def stable_task_payload(task: dict[str, Any]) -> str:
    """Canonical serialization used to fingerprint judge inputs."""

    return json.dumps(task, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
