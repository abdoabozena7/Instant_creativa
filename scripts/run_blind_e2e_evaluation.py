"""Run and score the frozen NG12 system on the separated blind evaluation set."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.engine import RetrievalEngine  # noqa: E402
from src.retrieval.generation import CITATION_PATTERN, generate_grounded_answer  # noqa: E402


DEFAULT_EVAL_DIR = PROJECT_ROOT / "data" / "eval"
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_EVAL_DIR / "blind_questions_v1.jsonl",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_EVAL_DIR / "blind_gold_v1.jsonl",
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=DEFAULT_EVAL_DIR / "evaluation_freeze.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--evidence-k", type=int, default=6)
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Reuse blind_run_v1.jsonl and regenerate reports without model calls.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze(path: Path) -> dict[str, Any]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    mismatches = []
    aggregate = hashlib.sha256()
    for relative, expected in freeze["files"].items():
        target = PROJECT_ROOT / relative
        actual = sha256(target)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(actual.encode("ascii"))
        if actual != expected["sha256"]:
            mismatches.append({"file": relative, "expected": expected["sha256"], "actual": actual})
    if aggregate.hexdigest() != freeze["architecture_sha256"] or mismatches:
        raise RuntimeError(
            "Frozen architecture changed before evaluation: " + json.dumps(mismatches)
        )
    return freeze


async def execute_questions(
    questions: list[dict[str, Any]],
    *,
    evidence_k: int,
    output_path: Path,
    freeze: dict[str, Any],
) -> list[dict[str, Any]]:
    """Execute questions without loading or receiving the gold labels."""

    engine = RetrievalEngine()
    rows: list[dict[str, Any]] = []
    partial_path = output_path.with_suffix(".partial.jsonl")
    for index, case in enumerate(questions, start=1):
        started = time.perf_counter()
        retrieval = await engine.search(
            case["question"], mode="hybrid", top_k=evidence_k
        )
        if retrieval["scope"]["status"] == "out_of_scope":
            generation = {
                "answer": retrieval["scope"]["message"],
                "model": None,
                "latency_ms": 0.0,
                "citation_validation": {
                    "passed": True,
                    "cited_evidence_ranks": [],
                    "invalid_evidence_ranks": [],
                    "available_evidence_count": 0,
                },
                "warnings": ["Generation skipped by deterministic scope guard."],
            }
        else:
            generation = await generate_grounded_answer(
                case["question"], retrieval["results"], engine.ollama
            )
        row = {
            **case,
            "architecture_sha256": freeze["architecture_sha256"],
            "retrieval": retrieval,
            "generation": generation,
            "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        rows.append(row)
        partial_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
            encoding="utf-8",
        )
        print(
            f"[{index:02d}/{len(questions)}] {case['case_id']} "
            f"scope={retrieval['scope']['status']} total={row['total_latency_ms']:.0f}ms"
        )

    output_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
        encoding="utf-8",
    )
    partial_path.unlink(missing_ok=True)
    return rows


def relevant(result: dict[str, Any], gold: dict[str, Any]) -> bool:
    recommendation_ids = set(gold.get("acceptable_recommendation_ids", []))
    if recommendation_ids and result.get("recommendation_id") in recommendation_ids:
        return True
    content_types = set(gold.get("acceptable_content_types", []))
    expected_sites = set(gold.get("expected_sites", []))
    return bool(content_types and expected_sites) and (
        result.get("content_type") in content_types
        and bool(expected_sites & set(result.get("cancer_sites", [])))
    )


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * proportion) - 1))
    return round(ordered[index], 2)


def split_claims(answer: str) -> list[dict[str, Any]]:
    claims = []
    for index, sentence in enumerate(SENTENCE_BOUNDARY.split(answer), start=1):
        sentence = sentence.strip()
        if not sentence:
            continue
        labels = [int(value) for value in CITATION_PATTERN.findall(sentence)]
        plain = CITATION_PATTERN.sub("", sentence).strip()
        claims.append(
            {
                "claim_id": f"C{index}",
                "text": plain,
                "cited_evidence_ranks": labels,
                "human_supported": None,
                "human_citation_entails_claim": None,
                "human_notes": "",
            }
        )
    return claims


def score_and_prepare_adjudication(
    rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    *,
    freeze: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    gold_by_id = {row["case_id"]: row for row in gold_rows}
    ranks: list[int | None] = []
    current_checks: list[bool] = []
    scope_checks: list[bool] = []
    correct_refusals: list[bool] = []
    false_refusals: list[bool] = []
    citation_checks: list[bool] = []
    total_latencies: list[float] = []
    generation_latencies: list[float] = []
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    diagnostics: list[dict[str, Any]] = []
    packets: list[dict[str, Any]] = []

    for row in rows:
        gold = gold_by_id[row["case_id"]]
        retrieval = row["retrieval"]
        generation = row["generation"]
        expected_scope = gold["expected_scope_status"]
        actual_scope = retrieval["scope"]["status"]
        scope_correct = actual_scope == expected_scope
        scope_checks.append(scope_correct)
        expected_refusal = gold["expected_behavior"] == "refuse"
        actual_refusal = actual_scope == "out_of_scope"
        if expected_refusal:
            correct_refusals.append(actual_refusal)
        else:
            false_refusals.append(actual_refusal)

        scored_retrieval = bool(
            gold.get("acceptable_recommendation_ids")
            or gold.get("acceptable_content_types")
        )
        rank = None
        if scored_retrieval:
            rank = next(
                (
                    index
                    for index, result in enumerate(retrieval["results"], start=1)
                    if relevant(result, gold)
                ),
                None,
            )
            ranks.append(rank)

        current_correct = None
        if gold.get("current_required") and rank:
            relevant_result = retrieval["results"][rank - 1]
            current_correct = (
                relevant_result["source_version"] == "2026_current"
                and relevant_result["authority_priority"] == "primary"
            )
            current_checks.append(current_correct)

        citation_valid = bool(generation["citation_validation"]["passed"])
        if not actual_refusal:
            citation_checks.append(citation_valid)
            generation_latencies.append(float(generation["latency_ms"]))
        total_latencies.append(float(row["total_latency_ms"]))
        diagnostic = {
            "case_id": row["case_id"],
            "scope_group": row["scope_group"],
            "category": row["category"],
            "expected_behavior": gold["expected_behavior"],
            "scope_correct": scope_correct,
            "retrieval_rank": rank,
            "current_guideline_correct": current_correct,
            "citation_labels_valid": citation_valid,
            "top_chunk_id": (
                retrieval["results"][0]["chunk_id"] if retrieval["results"] else None
            ),
            "top_recommendation_id": (
                retrieval["results"][0].get("recommendation_id")
                if retrieval["results"]
                else None
            ),
        }
        diagnostics.append(diagnostic)
        by_group[row["scope_group"]].append(diagnostic)

        evidence = [
            {
                "label": f"E{result['rank']}",
                "chunk_id": result["chunk_id"],
                "citation": result["citation"],
                "source_version": result["source_version"],
                "authority_priority": result["authority_priority"],
                "recommendation_id": result.get("recommendation_id"),
                "content_type": result["content_type"],
                "text": result["text"],
            }
            for result in retrieval["results"]
        ]
        packets.append(
            {
                "case_id": row["case_id"],
                "scope_group": row["scope_group"],
                "category": row["category"],
                "question": row["question"],
                "expected_behavior": gold["expected_behavior"],
                "required_concepts": gold["required_concepts"],
                "gold_rationale": gold["rationale"],
                "answer": generation["answer"],
                "claims": split_claims(generation["answer"]),
                "evidence": evidence,
                "human_answer_behavior_correct": None,
                "human_current_guideline_accuracy": None,
                "human_failure_types": [],
                "human_notes": "",
            }
        )

    retrieval_denominator = len(ranks)
    report = {
        "document": "NICE NG12",
        "evaluation_name": "blind_end_to_end_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture_sha256": freeze["architecture_sha256"],
        "architecture_frozen_at": freeze["frozen_at"],
        "questions": {
            "total": len(rows),
            "by_scope_group": dict(sorted(Counter(row["scope_group"] for row in rows).items())),
            "by_category": dict(sorted(Counter(row["category"] for row in rows).items())),
            "by_expected_behavior": dict(
                sorted(Counter(gold_by_id[row["case_id"]]["expected_behavior"] for row in rows).items())
            ),
        },
        "deterministic_metrics": {
            "scope_classification_accuracy": round(sum(scope_checks) / len(scope_checks), 4),
            "correct_refusal_rate": round(sum(correct_refusals) / len(correct_refusals), 4),
            "false_refusal_rate": round(sum(false_refusals) / len(false_refusals), 4),
            "retrieval_queries_scored": retrieval_denominator,
            "retrieval_recall_at_1": round(sum(rank == 1 for rank in ranks) / retrieval_denominator, 4),
            "retrieval_recall_at_3": round(
                sum(rank is not None and rank <= 3 for rank in ranks) / retrieval_denominator, 4
            ),
            "retrieval_recall_at_5": round(
                sum(rank is not None and rank <= 5 for rank in ranks) / retrieval_denominator, 4
            ),
            "retrieval_mrr_at_6": round(
                sum(1 / rank if rank else 0 for rank in ranks) / retrieval_denominator, 4
            ),
            "current_guideline_accuracy": round(sum(current_checks) / len(current_checks), 4),
            "citation_label_validity_rate": round(
                sum(citation_checks) / len(citation_checks), 4
            ),
            "latency_ms": {
                "end_to_end_p50": round(statistics.median(total_latencies), 2),
                "end_to_end_p95": percentile(total_latencies, 0.95),
                "generation_p50": round(statistics.median(generation_latencies), 2),
                "generation_p95": percentile(generation_latencies, 0.95),
            },
        },
        "semantic_metrics": {
            "status": "pending_human_adjudication",
            "citation_accuracy": None,
            "claim_support_rate": None,
            "unsupported_claim_rate": None,
            "answer_behavior_accuracy": None,
            "note": (
                "Valid citation labels do not prove entailment. Complete the claim-level adjudication "
                "packet before reporting these metrics."
            ),
        },
        "failures": {
            "scope": [item for item in diagnostics if not item["scope_correct"]],
            "retrieval_at_5": [
                item
                for item in diagnostics
                if (
                    gold_by_id[item["case_id"]].get("acceptable_recommendation_ids")
                    or gold_by_id[item["case_id"]].get("acceptable_content_types")
                )
                and (item["retrieval_rank"] is None or item["retrieval_rank"] > 5)
            ],
            "current_guideline": [
                item for item in diagnostics if item["current_guideline_correct"] is False
            ],
            "citation_labels": [
                item
                for item in diagnostics
                if not item["citation_labels_valid"]
                and item["expected_behavior"] != "refuse"
            ],
        },
        "by_scope_group": {
            group: {
                "cases": len(items),
                "scope_accuracy": round(sum(item["scope_correct"] for item in items) / len(items), 4),
                "retrieval_recall_at_5": _group_recall(items, gold_by_id, 5),
            }
            for group, items in sorted(by_group.items())
        },
        "case_diagnostics": diagnostics,
    }

    packets_path = output_dir / "adjudication_packets_v1.jsonl"
    packets_path.write_text(
        "".join(json.dumps(packet, ensure_ascii=False) + "\n" for packet in packets),
        encoding="utf-8",
    )
    _write_adjudication_csv(output_dir / "adjudication_template_v1.csv", packets)
    return report


def _group_recall(
    items: list[dict[str, Any]], gold_by_id: dict[str, dict[str, Any]], k: int
) -> float | None:
    scored = [
        item
        for item in items
        if gold_by_id[item["case_id"]].get("acceptable_recommendation_ids")
        or gold_by_id[item["case_id"]].get("acceptable_content_types")
    ]
    if not scored:
        return None
    return round(
        sum(item["retrieval_rank"] is not None and item["retrieval_rank"] <= k for item in scored)
        / len(scored),
        4,
    )


def _write_adjudication_csv(path: Path, packets: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "scope_group",
        "category",
        "expected_behavior",
        "question",
        "answer",
        "human_answer_behavior_correct",
        "human_total_claims",
        "human_supported_claims",
        "human_unsupported_claims",
        "human_citations_checked",
        "human_citations_entailed",
        "human_current_guideline_accuracy",
        "human_failure_types",
        "human_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for packet in packets:
            writer.writerow(
                {
                    "case_id": packet["case_id"],
                    "scope_group": packet["scope_group"],
                    "category": packet["category"],
                    "expected_behavior": packet["expected_behavior"],
                    "question": packet["question"],
                    "answer": packet["answer"],
                }
            )


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    freeze = verify_freeze(args.freeze.resolve())
    questions = load_jsonl(args.questions.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "blind_run_v1.jsonl"
    if args.score_only:
        rows = load_jsonl(run_path)
        if {row["case_id"] for row in rows} != {row["case_id"] for row in questions}:
            raise ValueError("Stored blind run does not match the question set")
    else:
        rows = await execute_questions(
            questions,
            evidence_k=args.evidence_k,
            output_path=run_path,
            freeze=freeze,
        )
    # Gold is loaded only after every model response has been persisted.
    gold_rows = load_jsonl(args.gold.resolve())
    report = score_and_prepare_adjudication(
        rows, gold_rows, freeze=freeze, output_dir=output_dir
    )
    report_path = output_dir / "blind_e2e_report_v1.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = asyncio.run(async_main(parse_args()))
    print(json.dumps({
        "questions": report["questions"],
        "deterministic_metrics": report["deterministic_metrics"],
        "semantic_metrics": report["semantic_metrics"],
        "failure_counts": {key: len(value) for key, value in report["failures"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
