"""Task construction and fail-closed consensus for automated semantic evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.judge_prompts import stable_task_payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def task_id(criterion: str, case_id: str, claim_id: str | None = None) -> str:
    suffix = f"_{claim_id}" if claim_id else ""
    return f"{case_id}{suffix}_{criterion}"


def build_tasks(
    packets: list[dict[str, Any]], provisional: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    verdict_by_case = {row["case_id"]: row for row in provisional}
    tasks: list[dict[str, Any]] = []
    deterministic: list[dict[str, Any]] = []
    for packet in packets:
        case_id = packet["case_id"]
        claims = verdict_by_case.get(case_id, {}).get("claims", [])
        normalized_claims = []
        for index, claim in enumerate(claims, start=1):
            claim_id = f"C{index}"
            cited_labels = list(claim.get("cited_labels") or [])
            all_evidence = packet["evidence"]
            cited_evidence = [
                item for item in all_evidence if item["label"] in set(cited_labels)
            ]
            normalized_claims.append(
                {"claim_id": claim_id, "claim": claim["claim"]}
            )
            support = {
                "task_id": task_id("claim_support", case_id, claim_id),
                "criterion": "claim_support",
                "case_id": case_id,
                "claim_id": claim_id,
                "scope_group": packet["scope_group"],
                "question": packet["question"],
                "claim": claim["claim"],
                "cited_labels": cited_labels,
                "all_evidence": all_evidence,
            }
            tasks.append(_fingerprint(support))
            if cited_labels:
                entailment = {
                    "task_id": task_id("citation_entailment", case_id, claim_id),
                    "criterion": "citation_entailment",
                    "case_id": case_id,
                    "claim_id": claim_id,
                    "scope_group": packet["scope_group"],
                    "claim": claim["claim"],
                    "cited_labels": cited_labels,
                    "cited_evidence": cited_evidence,
                }
                tasks.append(_fingerprint(entailment))
            else:
                deterministic.append(
                    {
                        "task_id": task_id(
                            "citation_entailment", case_id, claim_id
                        ),
                        "criterion": "citation_entailment",
                        "case_id": case_id,
                        "claim_id": claim_id,
                        "label": "NO",
                        "reason": "No citation label is attached to this claim.",
                        "source": "deterministic_missing_citation",
                    }
                )

        case_base = {
            "case_id": case_id,
            "scope_group": packet["scope_group"],
            "expected_behavior": packet["expected_behavior"],
            "question": packet["question"],
            "answer": packet["answer"],
            "all_evidence": packet["evidence"],
            "claims": normalized_claims,
        }
        for criterion in ("overreach", "completeness"):
            task = {
                "task_id": task_id(criterion, case_id),
                "criterion": criterion,
                **case_base,
            }
            tasks.append(_fingerprint(task))
        if packet["expected_behavior"] == "refuse":
            task = {
                "task_id": task_id("refusal_quality", case_id),
                "criterion": "refusal_quality",
                **case_base,
            }
            tasks.append(_fingerprint(task))
    return tasks, deterministic


def _fingerprint(task: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in task.items() if key != "input_sha256"}
    task["input_sha256"] = hashlib.sha256(
        stable_task_payload(payload).encode("utf-8")
    ).hexdigest()
    return task


def validate_judgment(task: dict[str, Any], judgment: dict[str, Any]) -> None:
    expected = (
        {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"}
        if task["criterion"] == "claim_support"
        else {"YES", "NO"}
    )
    if judgment.get("label") not in expected:
        raise ValueError(
            f"{task['task_id']}: invalid label {judgment.get('label')!r}"
        )
    if not isinstance(judgment.get("reason"), str):
        raise ValueError(f"{task['task_id']}: missing reason")
    if task["criterion"] == "overreach" and not isinstance(
        judgment.get("claim_ids"), list
    ):
        raise ValueError(f"{task['task_id']}: missing claim_ids")


def majority_label(labels: list[str]) -> str:
    counts = Counter(labels)
    if not counts:
        return "UNCERTAIN"
    label, count = counts.most_common(1)[0]
    return label if count >= 2 else "UNCERTAIN"


def consensus_label(label_a: str, label_b: str) -> tuple[str, bool]:
    if label_a == label_b and label_a != "UNCERTAIN":
        return label_a, True
    return "UNCERTAIN", False


def score_consensus(
    tasks: list[dict[str, Any]],
    deterministic: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    *,
    passes: int,
    judge_names: tuple[str, str],
    models: dict[str, str],
    architecture_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_by_id = {task["task_id"]: task for task in tasks}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        if row["task_id"] not in task_by_id:
            raise ValueError(f"Unknown task in judgments: {row['task_id']}")
        grouped[(row["task_id"], row["judge"])].append(row)

    consensus_rows: list[dict[str, Any]] = []
    agreement: list[bool] = []
    for task in tasks:
        per_judge = {}
        for judge in judge_names:
            rows = grouped.get((task["task_id"], judge), [])
            if len(rows) != passes:
                raise ValueError(
                    f"{task['task_id']} / {judge}: expected {passes} passes, found {len(rows)}"
                )
            ordered = sorted(rows, key=lambda item: item["pass"])
            per_judge[judge] = {
                "label": majority_label([item["label"] for item in ordered]),
                "passes": [item["label"] for item in ordered],
                "reasons": [item["reason"] for item in ordered],
            }
        final, agreed = consensus_label(
            per_judge[judge_names[0]]["label"],
            per_judge[judge_names[1]]["label"],
        )
        agreement.append(agreed)
        consensus_rows.append(
            {
                "task_id": task["task_id"],
                "criterion": task["criterion"],
                "case_id": task["case_id"],
                "claim_id": task.get("claim_id"),
                "input_sha256": task["input_sha256"],
                "judge_results": per_judge,
                "consensus_label": final,
                "judges_agree": agreed,
                "fail_closed": not _passes_criterion(task["criterion"], final),
            }
        )

    for row in deterministic:
        consensus_rows.append(
            {
                **row,
                "consensus_label": row["label"],
                "judges_agree": None,
                "fail_closed": True,
            }
        )

    by_criterion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in consensus_rows:
        by_criterion[row["criterion"]].append(row)

    support = by_criterion["claim_support"]
    entailment = by_criterion["citation_entailment"]
    overreach = by_criterion["overreach"]
    completeness = by_criterion["completeness"]
    refusal = by_criterion["refusal_quality"]
    failures = [row for row in consensus_rows if row["fail_closed"]]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_automated_multi_judge",
        "evaluation_name": "automated_multi_judge_v1",
        "architecture_sha256": architecture_sha256,
        "judges": {
            judge: {"model": models[judge], "passes": passes}
            for judge in judge_names
        },
        "disclaimer": (
            "Automated groundedness evaluation using two LLM judges. "
            "This is not clinical validation."
        ),
        "counts": {
            "llm_tasks": len(tasks),
            "deterministic_missing_citation_tasks": len(deterministic),
            "model_calls": len(judgments),
            "consensus_decisions": len(consensus_rows),
        },
        "metrics": {
            "judge_agreement_rate": _rate(agreement),
            "claim_support_rate": _pass_rate(support),
            "unsupported_claim_rate": _fail_rate(support),
            "citation_entailment_rate": _pass_rate(entailment),
            "citation_coverage_rate": round(
                (len(entailment) - len(deterministic)) / len(entailment), 4
            ),
            "overreach_free_rate": _pass_rate(overreach),
            "answer_completeness_rate": _pass_rate(completeness),
            "refusal_quality_rate": _pass_rate(refusal),
        },
        "labels_by_criterion": {
            criterion: dict(
                sorted(Counter(row["consensus_label"] for row in rows).items())
            )
            for criterion, rows in sorted(by_criterion.items())
        },
        "failure_counts_by_criterion": dict(
            sorted(Counter(row["criterion"] for row in failures).items())
        ),
        "disagreements": [
            {
                "task_id": row["task_id"],
                "criterion": row["criterion"],
                "case_id": row["case_id"],
                "claim_id": row.get("claim_id"),
            }
            for row in consensus_rows
            if row["judges_agree"] is False
        ],
    }
    return report, consensus_rows


def score_hackathon(
    tasks: list[dict[str, Any]],
    deterministic: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    *,
    models: dict[str, str],
    architecture_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score one Gemini pass, escalating only primary failures to GPT-OSS."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        grouped[(row["task_id"], row["judge"])].append(row)

    consensus_rows: list[dict[str, Any]] = []
    agreement: list[bool] = []
    escalated = 0
    for task in tasks:
        primary_rows = grouped.get((task["task_id"], "gemini"), [])
        if len(primary_rows) != 1:
            raise ValueError(
                f"{task['task_id']} / gemini: expected 1 pass, found {len(primary_rows)}"
            )
        primary = primary_rows[0]
        primary_passes = _passes_criterion(task["criterion"], primary["label"])
        final = primary["label"]
        agreed: bool | None = None
        secondary: dict[str, Any] | None = None
        if not primary_passes:
            escalated += 1
            secondary_rows = grouped.get((task["task_id"], "gpt_oss"), [])
            if len(secondary_rows) != 1:
                raise ValueError(
                    f"{task['task_id']} / gpt_oss: expected 1 escalation pass, "
                    f"found {len(secondary_rows)}"
                )
            secondary = secondary_rows[0]
            final, agreed = consensus_label(primary["label"], secondary["label"])
            agreement.append(agreed)
        consensus_rows.append(
            {
                "task_id": task["task_id"],
                "criterion": task["criterion"],
                "case_id": task["case_id"],
                "claim_id": task.get("claim_id"),
                "input_sha256": task["input_sha256"],
                "primary_result": {
                    "judge": "gemini",
                    "label": primary["label"],
                    "reason": primary["reason"],
                },
                "secondary_result": (
                    {
                        "judge": "gpt_oss",
                        "label": secondary["label"],
                        "reason": secondary["reason"],
                    }
                    if secondary
                    else None
                ),
                "escalated": secondary is not None,
                "consensus_label": final,
                "judges_agree": agreed,
                "fail_closed": not _passes_criterion(task["criterion"], final),
            }
        )

    for row in deterministic:
        consensus_rows.append(
            {
                **row,
                "consensus_label": row["label"],
                "judges_agree": None,
                "escalated": False,
                "fail_closed": True,
            }
        )

    by_criterion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in consensus_rows:
        by_criterion[row["criterion"]].append(row)
    failures = [row for row in consensus_rows if row["fail_closed"]]
    entailment = by_criterion["citation_entailment"]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_automated_hackathon_judge",
        "evaluation_name": "automated_hackathon_judge_v1",
        "architecture_sha256": architecture_sha256,
        "judges": {
            "gemini": {"model": models["gemini"], "passes": 1, "role": "primary"},
            "gpt_oss": {
                "model": models["gpt_oss"],
                "passes": 1,
                "role": "conditional_second_judge",
            },
        },
        "disclaimer": (
            "Automated groundedness evaluation with conditional second-judge review. "
            "This is not clinical validation."
        ),
        "counts": {
            "llm_tasks": len(tasks),
            "deterministic_missing_citation_tasks": len(deterministic),
            "model_calls": len(judgments),
            "primary_model_calls": len(tasks),
            "secondary_model_calls": escalated,
            "escalated_tasks": escalated,
            "consensus_decisions": len(consensus_rows),
        },
        "metrics": {
            "judge_agreement_rate": _rate(agreement),
            "claim_support_rate": _pass_rate(by_criterion["claim_support"]),
            "unsupported_claim_rate": _fail_rate(by_criterion["claim_support"]),
            "citation_entailment_rate": _pass_rate(entailment),
            "citation_coverage_rate": round(
                (len(entailment) - len(deterministic)) / len(entailment), 4
            ),
            "overreach_free_rate": _pass_rate(by_criterion["overreach"]),
            "answer_completeness_rate": _pass_rate(by_criterion["completeness"]),
            "refusal_quality_rate": _pass_rate(by_criterion["refusal_quality"]),
        },
        "labels_by_criterion": {
            criterion: dict(
                sorted(Counter(row["consensus_label"] for row in rows).items())
            )
            for criterion, rows in sorted(by_criterion.items())
        },
        "failure_counts_by_criterion": dict(
            sorted(Counter(row["criterion"] for row in failures).items())
        ),
        "disagreements": [
            {
                "task_id": row["task_id"],
                "criterion": row["criterion"],
                "case_id": row["case_id"],
                "claim_id": row.get("claim_id"),
            }
            for row in consensus_rows
            if row["judges_agree"] is False
        ],
    }
    return report, consensus_rows


def _passes_criterion(criterion: str, label: str) -> bool:
    if criterion == "claim_support":
        return label == "SUPPORTED"
    if criterion == "overreach":
        return label == "NO"
    return label == "YES"


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _pass_rate(rows: list[dict[str, Any]]) -> float | None:
    return _rate([not row["fail_closed"] for row in rows])


def _fail_rate(rows: list[dict[str, Any]]) -> float | None:
    value = _pass_rate(rows)
    return round(1 - value, 4) if value is not None else None
