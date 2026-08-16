from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.judge_clients import GeminiJudge
from src.evaluation.judge_prompts import prompt_and_schema
from src.evaluation.multi_judge import (
    build_tasks,
    load_jsonl,
    score_consensus,
    score_hackathon,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "data" / "eval"


def prepared() -> tuple[list[dict], list[dict]]:
    return build_tasks(
        load_jsonl(EVAL_DIR / "adjudication_packets_v1.jsonl"),
        load_jsonl(EVAL_DIR / "provisional_claim_adjudication_v1.jsonl"),
    )


def test_gemini_key_must_come_from_secure_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        GeminiJudge()


def test_multi_judge_task_plan_is_decomposed_and_complete() -> None:
    tasks, deterministic = prepared()
    criteria = [task["criterion"] for task in tasks]
    assert len(tasks) == 334
    assert len(deterministic) == 27
    assert criteria.count("claim_support") == 134
    assert criteria.count("citation_entailment") == 107
    assert criteria.count("overreach") == 44
    assert criteria.count("completeness") == 44
    assert criteria.count("refusal_quality") == 5
    assert all(row["label"] == "NO" for row in deterministic)
    assert all(row["source"] == "deterministic_missing_citation" for row in deterministic)


def test_absence_claim_prompt_explicitly_forbids_absence_inference() -> None:
    tasks, _ = prepared()
    absence = next(task for task in tasks if task["task_id"] == "BL-L02_C3_claim_support")
    prompt, schema = prompt_and_schema(absence)
    assert "Failure to retrieve an alternative is not proof that none exists" in prompt
    assert "CLAIM SUPPORT ONLY" in prompt
    assert "SUPPORTED" in schema["properties"]["label"]["enum"]


def test_consensus_is_fail_closed_on_cross_judge_disagreement() -> None:
    tasks, deterministic = prepared()
    judgments = []
    disagreement_task = next(
        task for task in tasks if task["criterion"] == "claim_support"
    )
    for task in tasks:
        passing_label = {
            "claim_support": "SUPPORTED",
            "citation_entailment": "YES",
            "overreach": "NO",
            "completeness": "YES",
            "refusal_quality": "YES",
        }[task["criterion"]]
        for judge in ("gemini", "gpt_oss"):
            label = passing_label
            if task["task_id"] == disagreement_task["task_id"] and judge == "gpt_oss":
                label = "UNSUPPORTED"
            for pass_number in range(1, 4):
                judgments.append(
                    {
                        "task_id": task["task_id"],
                        "judge": judge,
                        "pass": pass_number,
                        "label": label,
                        "reason": "test",
                    }
                )

    freeze = json.loads(
        (EVAL_DIR / "evaluation_freeze.json").read_text(encoding="utf-8")
    )
    report, consensus = score_consensus(
        tasks,
        deterministic,
        judgments,
        passes=3,
        judge_names=("gemini", "gpt_oss"),
        models={"gemini": "test-a", "gpt_oss": "test-b"},
        architecture_sha256=freeze["architecture_sha256"],
    )
    disputed = next(
        row for row in consensus if row["task_id"] == disagreement_task["task_id"]
    )
    assert disputed["consensus_label"] == "UNCERTAIN"
    assert disputed["fail_closed"] is True
    assert report["metrics"]["judge_agreement_rate"] < 1
    assert report["metrics"]["claim_support_rate"] == 0.9925
    assert report["metrics"]["citation_entailment_rate"] == 0.7985
    assert report["disclaimer"].endswith("not clinical validation.")


def test_hackathon_mode_escalates_only_primary_failures_and_fails_closed() -> None:
    tasks, deterministic = prepared()
    judgments = []
    disputed_task = next(task for task in tasks if task["criterion"] == "overreach")
    pass_labels = {
        "claim_support": "SUPPORTED",
        "citation_entailment": "YES",
        "overreach": "NO",
        "completeness": "YES",
        "refusal_quality": "YES",
    }
    for task in tasks:
        label = pass_labels[task["criterion"]]
        if task["task_id"] == disputed_task["task_id"]:
            label = "YES"
        judgments.append(
            {
                "task_id": task["task_id"],
                "judge": "gemini",
                "pass": 1,
                "label": label,
                "reason": "primary test",
            }
        )
    judgments.append(
        {
            "task_id": disputed_task["task_id"],
            "judge": "gpt_oss",
            "pass": 1,
            "label": "NO",
            "reason": "secondary test",
        }
    )

    freeze = json.loads(
        (EVAL_DIR / "evaluation_freeze.json").read_text(encoding="utf-8")
    )
    report, consensus = score_hackathon(
        tasks,
        deterministic,
        judgments,
        models={"gemini": "test-a", "gpt_oss": "test-b"},
        architecture_sha256=freeze["architecture_sha256"],
    )
    disputed = next(
        row for row in consensus if row["task_id"] == disputed_task["task_id"]
    )
    assert report["counts"]["primary_model_calls"] == 334
    assert report["counts"]["secondary_model_calls"] == 1
    assert disputed["consensus_label"] == "UNCERTAIN"
    assert disputed["fail_closed"] is True
    assert disputed["escalated"] is True
