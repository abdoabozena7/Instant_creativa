from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.score_human_adjudication import score_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "data" / "eval"


def load_jsonl(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (EVAL_DIR / name).read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_completed_blind_run_preserves_its_frozen_architecture_identity() -> None:
    freeze = json.loads(
        (EVAL_DIR / "evaluation_freeze.json").read_text(encoding="utf-8")
    )
    run = load_jsonl("blind_run_v1.jsonl")
    assert len(run) == 44
    assert {row["architecture_sha256"] for row in run} == {
        freeze["architecture_sha256"]
    }


def test_blind_questions_are_balanced_and_do_not_contain_gold_labels() -> None:
    questions = load_jsonl("blind_questions_v1.jsonl")
    assert len(questions) == 44
    assert len({row["case_id"] for row in questions}) == 44
    assert Counter(row["scope_group"] for row in questions) == {
        "lung": 11,
        "colorectal": 11,
        "upper_gi": 11,
        "bladder_renal": 11,
    }
    forbidden = {
        "expected_behavior",
        "expected_scope_status",
        "acceptable_recommendation_ids",
        "required_concepts",
        "rationale",
    }
    assert all(forbidden.isdisjoint(row) for row in questions)


def test_question_gold_and_adjudication_artifacts_have_identical_case_sets() -> None:
    questions = load_jsonl("blind_questions_v1.jsonl")
    gold = load_jsonl("blind_gold_v1.jsonl")
    packets = load_jsonl("adjudication_packets_v1.jsonl")
    question_ids = {row["case_id"] for row in questions}
    assert {row["case_id"] for row in gold} == question_ids
    assert {row["case_id"] for row in packets} == question_ids
    with (EVAL_DIR / "adjudication_template_v1.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        csv_rows = list(csv.DictReader(stream))
    assert {row["case_id"] for row in csv_rows} == question_ids


def test_blind_report_preserves_known_deterministic_results_and_failures() -> None:
    report = json.loads(
        (EVAL_DIR / "blind_e2e_report_v1.json").read_text(encoding="utf-8")
    )
    metrics = report["deterministic_metrics"]
    assert report["questions"]["total"] == 44
    assert metrics["scope_classification_accuracy"] == 0.9773
    assert metrics["correct_refusal_rate"] == 0.8
    assert metrics["false_refusal_rate"] == 0.0
    assert metrics["retrieval_recall_at_5"] == 0.973
    assert metrics["current_guideline_accuracy"] == 1.0
    assert metrics["citation_label_validity_rate"] == 0.8
    assert [row["case_id"] for row in report["failures"]["scope"]] == ["BL-U10"]
    assert [row["case_id"] for row in report["failures"]["retrieval_at_5"]] == [
        "BL-L03"
    ]
    assert len(report["failures"]["citation_labels"]) == 8


def test_post_fix_v2_rejects_gall_bladder_without_regressing_retrieval() -> None:
    freeze = json.loads(
        (EVAL_DIR / "evaluation_freeze_v2.json").read_text(encoding="utf-8")
    )
    run = load_jsonl("blind_run_v2.jsonl")
    report = json.loads(
        (EVAL_DIR / "blind_e2e_report_v2.json").read_text(encoding="utf-8")
    )
    metrics = report["deterministic_metrics"]
    gall_bladder = next(row for row in run if row["case_id"] == "BL-U10")

    assert {row["architecture_sha256"] for row in run} == {
        freeze["architecture_sha256"]
    }
    assert gall_bladder["retrieval"]["scope"]["status"] == "out_of_scope"
    assert gall_bladder["retrieval"]["results"] == []
    assert metrics["scope_classification_accuracy"] == 1.0
    assert metrics["correct_refusal_rate"] == 1.0
    assert metrics["false_refusal_rate"] == 0.0
    assert metrics["retrieval_recall_at_1"] == 0.7568
    assert metrics["retrieval_recall_at_5"] == 0.973
    assert metrics["retrieval_mrr_at_6"] == 0.8559
    assert metrics["current_guideline_accuracy"] == 1.0
    assert report["failures"]["scope"] == []


def test_answerability_fix_v4_preserves_retrieval_and_canonical_citations() -> None:
    freeze = json.loads(
        (EVAL_DIR / "evaluation_freeze_v4.json").read_text(encoding="utf-8")
    )
    run = load_jsonl("blind_run_v4.jsonl")
    report = json.loads(
        (EVAL_DIR / "blind_e2e_report_v4.json").read_text(encoding="utf-8")
    )
    metrics = report["deterministic_metrics"]

    assert len(run) == 44
    assert {row["architecture_sha256"] for row in run} == {
        freeze["architecture_sha256"]
    }
    assert metrics["scope_classification_accuracy"] == 1.0
    assert metrics["correct_refusal_rate"] == 1.0
    assert metrics["false_refusal_rate"] == 0.0
    assert metrics["retrieval_recall_at_1"] == 0.7568
    assert metrics["retrieval_recall_at_5"] == 0.973
    assert metrics["retrieval_mrr_at_6"] == 0.8604
    assert metrics["current_guideline_accuracy"] == 1.0
    assert metrics["citation_label_validity_rate"] == 1.0
    assert report["failures"]["scope"] == []
    assert report["failures"]["citation_labels"] == []
    assert report["semantic_metrics"]["status"] == "not_run"


def test_vague_assessment_fix_v6_preserves_all_deterministic_metrics() -> None:
    freeze = json.loads(
        (EVAL_DIR / "evaluation_freeze_v6.json").read_text(encoding="utf-8")
    )
    run = load_jsonl("blind_run_v6.jsonl")
    report = json.loads(
        (EVAL_DIR / "blind_e2e_report_v6.json").read_text(encoding="utf-8")
    )
    metrics = report["deterministic_metrics"]

    assert len(run) == 44
    assert {row["architecture_sha256"] for row in run} == {
        freeze["architecture_sha256"]
    }
    assert metrics["scope_classification_accuracy"] == 1.0
    assert metrics["correct_refusal_rate"] == 1.0
    assert metrics["false_refusal_rate"] == 0.0
    assert metrics["retrieval_queries_scored"] == 37
    assert metrics["retrieval_recall_at_1"] == 0.7568
    assert metrics["retrieval_recall_at_5"] == 0.973
    assert metrics["retrieval_mrr_at_6"] == 0.8604
    assert metrics["current_guideline_accuracy"] == 1.0
    assert metrics["citation_label_validity_rate"] == 1.0
    assert report["failures"]["scope"] == []
    assert report["failures"]["citation_labels"] == []
    assert report["semantic_metrics"]["status"] == "not_run"


def test_instruction_safety_fix_v8_preserves_all_deterministic_metrics() -> None:
    freeze = json.loads(
        (EVAL_DIR / "evaluation_freeze_v8.json").read_text(encoding="utf-8")
    )
    run = load_jsonl("blind_run_v8.jsonl")
    report = json.loads(
        (EVAL_DIR / "blind_e2e_report_v8.json").read_text(encoding="utf-8")
    )
    metrics = report["deterministic_metrics"]

    assert "src/retrieval/query_safety.py" in freeze["files"]
    assert len(run) == 44
    assert {row["architecture_sha256"] for row in run} == {
        freeze["architecture_sha256"]
    }
    assert metrics["scope_classification_accuracy"] == 1.0
    assert metrics["correct_refusal_rate"] == 1.0
    assert metrics["false_refusal_rate"] == 0.0
    assert metrics["retrieval_queries_scored"] == 37
    assert metrics["retrieval_recall_at_1"] == 0.7568
    assert metrics["retrieval_recall_at_5"] == 0.973
    assert metrics["retrieval_mrr_at_6"] == 0.8604
    assert metrics["current_guideline_accuracy"] == 1.0
    assert metrics["citation_label_validity_rate"] == 1.0
    assert report["failures"]["scope"] == []
    assert report["failures"]["citation_labels"] == []
    assert report["semantic_metrics"]["status"] == "not_run"


def test_emergency_and_claim_coverage_v12_artifacts_match_current_freeze() -> None:
    freeze = json.loads(
        (EVAL_DIR / "evaluation_freeze_v12.json").read_text(encoding="utf-8")
    )
    run = load_jsonl("blind_run_v12.jsonl")
    report = json.loads(
        (EVAL_DIR / "blind_e2e_report_v12.json").read_text(encoding="utf-8")
    )
    metrics = report["deterministic_metrics"]

    assert "src/retrieval/emergency_guard.py" in freeze["files"]
    assert len(run) == 44
    assert {row["architecture_sha256"] for row in run} == {
        freeze["architecture_sha256"]
    }
    assert report["evaluation_name"] == "blind_end_to_end_v12"
    assert metrics["scope_classification_accuracy"] == 1.0
    assert metrics["correct_refusal_rate"] == 1.0
    assert metrics["false_refusal_rate"] == 0.0
    assert metrics["retrieval_recall_at_5"] == 0.973
    assert metrics["retrieval_precision_at_3"] == 0.3423
    assert metrics["retrieval_precision_at_5"] == 0.2108
    assert metrics["current_guideline_accuracy"] == 1.0
    assert metrics["citation_label_validity_rate"] == 1.0
    assert metrics["claim_citation_coverage_rate"] == 0.985
    assert metrics["citation_release_pass_rate"] == 0.9487
    assert len(report["failures"]["claim_citation_coverage"]) == 2
    assert report["semantic_metrics"]["status"] == "not_run"


def test_semantic_scores_are_not_promoted_before_human_adjudication() -> None:
    report = json.loads(
        (EVAL_DIR / "blind_e2e_report_v1.json").read_text(encoding="utf-8")
    )
    semantic = report["semantic_metrics"]
    assert semantic["status"] == "pending_human_adjudication"
    assert semantic["citation_accuracy"] is None
    assert semantic["claim_support_rate"] is None
    assert semantic["unsupported_claim_rate"] is None
    provisional = semantic["provisional_model_assisted"]
    assert provisional["status"] == "provisional_model_assisted_pending_human_review"
    assert provisional["judge_model"] == "gpt-oss:120b-cloud"
    assert "same model" in provisional["independence_limitation"].lower()
    assert len(provisional["human_review_queue"]) == 24


def test_human_scorer_uses_only_resolved_final_judgments() -> None:
    cases = [
        {
            "Case ID": f"CASE-{index:02d}",
            "Final behavior correct": "TRUE" if index < 43 else "FALSE",
            "Final current accuracy": "TRUE" if index < 40 else "NOT_APPLICABLE",
            "Final claim list complete": "TRUE",
            "Final failure types": "overreach" if index == 43 else "",
        }
        for index in range(44)
    ]
    claims = [
        {
            "Claim key": "CASE-00-C1",
            "Cited labels": "E1",
            "Final supported": "TRUE",
            "Final citation entails": "TRUE",
            "Final severity": "NONE",
        },
        {
            "Claim key": "CASE-00-C2",
            "Cited labels": "E2",
            "Final supported": "FALSE",
            "Final citation entails": "FALSE",
            "Final severity": "MAJOR",
        },
    ]
    metrics = score_rows(cases, claims)
    assert metrics["answer_behavior_accuracy"] == 0.9773
    assert metrics["current_guideline_accuracy_human"] == 1.0
    assert metrics["claim_support_rate"] == 0.5
    assert metrics["unsupported_claim_rate"] == 0.5
    assert metrics["citation_accuracy"] == 0.5
    assert metrics["failure_type_counts"] == {"overreach": 1}


def test_human_scorer_fails_closed_on_incomplete_claim_list() -> None:
    cases = [
        {
            "Case ID": f"CASE-{index:02d}",
            "Final behavior correct": "TRUE",
            "Final current accuracy": "NOT_APPLICABLE",
            "Final claim list complete": "FALSE" if index == 0 else "TRUE",
            "Final failure types": "",
        }
        for index in range(44)
    ]
    claims = [
        {
            "Claim key": "CASE-00-C1",
            "Cited labels": "",
            "Final supported": "TRUE",
            "Final citation entails": "NOT_APPLICABLE",
            "Final severity": "NONE",
        }
    ]
    with pytest.raises(ValueError, match="claim list is incomplete"):
        score_rows(cases, claims)


def test_human_adjudication_workbook_inputs_are_versioned() -> None:
    required_inputs = [
        EVAL_DIR / "adjudication_packets_v1.jsonl",
        EVAL_DIR / "provisional_claim_adjudication_v1.jsonl",
        EVAL_DIR / "blind_e2e_report_v1.json",
        PROJECT_ROOT / "scripts" / "build_human_adjudication_workbook.mjs",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required_inputs)
