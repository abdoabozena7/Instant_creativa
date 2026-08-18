"""Explain strict non-gold Top-K results and the metric's corpus-label ceiling."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "data" / "eval"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def relevant(result: dict[str, Any], gold: dict[str, Any]) -> bool:
    if result.get("record_id") in gold.get("acceptable_record_ids", []):
        return True
    if result.get("recommendation_id") in gold.get(
        "acceptable_recommendation_ids", []
    ):
        return True
    expected_types = set(gold.get("acceptable_content_types", []))
    expected_sites = set(gold.get("expected_sites", []))
    return bool(expected_types and expected_sites) and (
        result.get("content_type") in expected_types
        and bool(expected_sites & set(result.get("cancer_sites", [])))
    )


def non_gold_reason(
    result: dict[str, Any], gold: dict[str, Any], seen_units: set[tuple[Any, ...]]
) -> str:
    unit = (
        result.get("record_id"),
        result.get("recommendation_id"),
        result.get("content_type"),
        result.get("text"),
    )
    if unit in seen_units:
        return "duplicate_unit"

    acceptable_ids = set(gold.get("acceptable_recommendation_ids", []))
    expected_sites = set(gold.get("expected_sites", []))
    expected_types = set(gold.get("acceptable_content_types", []))
    result_sites = set(result.get("cancer_sites", []))

    if acceptable_ids & set(result.get("related_recommendation_ids", [])):
        return "close_context_referencing_gold_id"
    if expected_sites and result_sites and not (expected_sites & result_sites):
        return "wrong_cancer_site"
    if expected_types and result.get("content_type") not in expected_types:
        return "wrong_content_type"
    if (
        acceptable_ids
        and expected_sites & result_sites
        and result.get("content_type") in {"recommendation", "symptom_table"}
    ):
        return "same_site_alternative_unit"
    if expected_sites & result_sites:
        return "same_site_context"
    if not result_sites:
        return "general_context"
    return "other"


def analyze(
    run_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    gold_by_id = {row["case_id"]: row for row in gold_rows}
    reason_counts: Counter[str] = Counter()
    cases: list[dict[str, Any]] = []
    retrieved_gold_count = 0
    corpus_gold_capacity = 0

    for row in run_rows:
        gold = gold_by_id[row["case_id"]]
        if not (
            gold.get("acceptable_recommendation_ids")
            or gold.get("acceptable_content_types")
            or gold.get("acceptable_record_ids")
        ):
            continue

        corpus_gold_count = sum(relevant(chunk, gold) for chunk in chunks)
        case_capacity = min(corpus_gold_count, top_k)
        corpus_gold_capacity += case_capacity
        seen_units: set[tuple[Any, ...]] = set()
        non_gold_results = []
        case_gold_count = 0

        for result in row["retrieval"]["results"][:top_k]:
            if relevant(result, gold):
                case_gold_count += 1
            else:
                reason = non_gold_reason(result, gold, seen_units)
                reason_counts[reason] += 1
                non_gold_results.append(
                    {
                        "rank": result["rank"],
                        "chunk_id": result["chunk_id"],
                        "recommendation_id": result.get("recommendation_id"),
                        "content_type": result["content_type"],
                        "cancer_sites": result.get("cancer_sites", []),
                        "reason": reason,
                    }
                )
            seen_units.add(
                (
                    result.get("record_id"),
                    result.get("recommendation_id"),
                    result.get("content_type"),
                    result.get("text"),
                )
            )

        retrieved_gold_count += case_gold_count
        cases.append(
            {
                "case_id": row["case_id"],
                "category": row.get("category"),
                "gold_results_retrieved": case_gold_count,
                "gold_results_available_in_corpus": corpus_gold_count,
                "gold_capacity_at_k": case_capacity,
                "missing_gold_slots_below_ceiling": case_capacity - case_gold_count,
                "non_gold_results": non_gold_results,
            }
        )

    denominator = len(cases) * top_k
    non_gold_count = denominator - retrieved_gold_count
    unavoidable_non_gold = denominator - corpus_gold_capacity
    avoidable_missing_gold = corpus_gold_capacity - retrieved_gold_count
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "top_k": top_k,
        "scored_queries": len(cases),
        "strict_precision": round(retrieved_gold_count / denominator, 4),
        "strict_precision_ceiling_from_labels_and_corpus": round(
            corpus_gold_capacity / denominator, 4
        ),
        "ceiling_adjusted_precision": round(
            retrieved_gold_count / corpus_gold_capacity, 4
        ),
        "counts": {
            "retrieved_results": denominator,
            "gold_results": retrieved_gold_count,
            "non_gold_results": non_gold_count,
            "structurally_unavoidable_non_gold_results": unavoidable_non_gold,
            "avoidable_missing_gold_slots": avoidable_missing_gold,
        },
        "non_gold_reason_counts": dict(reason_counts.most_common()),
        "cases_below_ceiling": [
            case for case in cases if case["missing_gold_slots_below_ceiling"] > 0
        ],
        "cases": cases,
        "interpretation": (
            "The strict ceiling counts how many labeled gold chunks actually exist in the "
            "corpus for each query, capped at K. Non-gold results beyond that capacity "
            "cannot be removed by reranking under the current labels."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=EVAL_DIR / "blind_run_v12.jsonl")
    parser.add_argument("--gold", type=Path, default=EVAL_DIR / "blind_gold_v1.jsonl")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=PROJECT_ROOT / "data" / "parsed" / "chunks.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_DIR / "retrieval_noise_diagnostics_v12.json",
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    report = analyze(
        load_jsonl(args.run.resolve()),
        load_jsonl(args.gold.resolve()),
        load_jsonl(args.chunks.resolve()),
        top_k=args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "strict_precision": report["strict_precision"],
                "strict_precision_ceiling": report[
                    "strict_precision_ceiling_from_labels_and_corpus"
                ],
                "ceiling_adjusted_precision": report["ceiling_adjusted_precision"],
                "counts": report["counts"],
                "non_gold_reason_counts": report["non_gold_reason_counts"],
                "cases_below_ceiling": [
                    case["case_id"] for case in report["cases_below_ceiling"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
