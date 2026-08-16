"""Evaluate BM25, dense, and hybrid retrieval on a labeled NG12 query set."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.engine import RetrievalEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "retrieval_cases.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "retrieval_metrics.json",
    )
    return parser.parse_args()


def relevant(result: dict[str, Any], case: dict[str, Any]) -> bool:
    if result.get("record_id") in case.get("expected_record_ids", []):
        return True
    if result.get("recommendation_id") in case.get("expected_recommendation_ids", []):
        return True
    expected_types = set(case.get("expected_content_types", []))
    expected_sites = set(case.get("expected_sites", []))
    return bool(expected_types and expected_sites) and (
        result.get("content_type") in expected_types
        and bool(expected_sites & set(result.get("cancer_sites", [])))
    )


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * proportion) - 1))
    return round(ordered[index], 2)


async def evaluate(cases_path: Path) -> dict[str, Any]:
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    engine = RetrievalEngine()
    if not engine.dense_available:
        raise FileNotFoundError(
            "Dense index is missing. Run scripts/build_retrieval_index.py before evaluation."
        )
    evaluations: dict[str, Any] = {}
    for mode in ["bm25", "dense", "hybrid"]:
        ranks: list[int | None] = []
        latencies: list[float] = []
        authority_checks: list[bool] = []
        category_ranks: dict[str, list[int | None]] = defaultdict(list)
        diagnostics = []
        out_of_scope_checks: list[bool] = []
        for case in cases:
            response = await engine.search(case["query"], mode=mode, top_k=10)
            latencies.append(response["latency_ms"])
            if case.get("out_of_scope"):
                passed = response["scope"]["status"] == "out_of_scope" and not response["results"]
                out_of_scope_checks.append(passed)
                diagnostics.append(
                    {"case_id": case["case_id"], "out_of_scope_passed": passed, "rank": None}
                )
                continue
            rank = next(
                (index for index, result in enumerate(response["results"], start=1) if relevant(result, case)),
                None,
            )
            ranks.append(rank)
            category_ranks[case["category"]].append(rank)
            if case.get("authority_required"):
                top = response["results"][0] if response["results"] else None
                authority_checks.append(
                    bool(top and relevant(top, case) and top["source_version"] == "2026_current")
                )
            diagnostics.append(
                {
                    "case_id": case["case_id"],
                    "rank": rank,
                    "top_chunk_id": response["results"][0]["chunk_id"] if response["results"] else None,
                    "top_recommendation_id": (
                        response["results"][0].get("recommendation_id") if response["results"] else None
                    ),
                    "latency_ms": response["latency_ms"],
                }
            )

        evaluations[mode] = {
            "queries_evaluated": len(cases),
            "in_scope_queries": len(ranks),
            "out_of_scope_queries": len(out_of_scope_checks),
            "recall_at_1": round(sum(rank == 1 for rank in ranks) / len(ranks), 4),
            "recall_at_3": round(sum(rank is not None and rank <= 3 for rank in ranks) / len(ranks), 4),
            "recall_at_5": round(sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks), 4),
            "mrr_at_10": round(sum(1 / rank if rank else 0 for rank in ranks) / len(ranks), 4),
            "canonical_top1_accuracy": round(sum(authority_checks) / len(authority_checks), 4),
            "out_of_scope_refusal_accuracy": round(
                sum(out_of_scope_checks) / len(out_of_scope_checks), 4
            ),
            "latency_ms": {
                "p50": round(statistics.median(latencies), 2),
                "p95": percentile(latencies, 0.95),
            },
            "recall_at_5_by_category": {
                category: round(
                    sum(rank is not None and rank <= 5 for rank in category_values)
                    / len(category_values),
                    4,
                )
                for category, category_values in sorted(category_ranks.items())
            },
            "failures_at_5": [
                item for item in diagnostics if item.get("rank") is not None and item["rank"] > 5
            ]
            + [item for item in diagnostics if "rank" in item and item["rank"] is None and "out_of_scope_passed" not in item],
            "case_diagnostics": diagnostics,
        }
    best_mode = max(
        evaluations,
        key=lambda mode: (
            evaluations[mode]["recall_at_5"],
            evaluations[mode]["canonical_top1_accuracy"],
            evaluations[mode]["mrr_at_10"],
        ),
    )
    return {
        "document": "NICE NG12",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_set": {
            "path": str(cases_path.relative_to(PROJECT_ROOT)),
            "queries": len(cases),
            "by_category": dict(sorted(Counter(case["category"] for case in cases).items())),
        },
        "embedding_model": engine.ollama.embedding_model,
        "index": {
            "type": "exact cosine over normalized NumPy matrix",
            "chunks": len(engine.chunks),
            "dimensions": int(engine.embeddings.shape[1]),
        },
        "modes": evaluations,
        "recommended_mode": best_mode,
        "selection_rule": (
            "Maximize Recall@5, then canonical top-1 accuracy, then MRR@10. "
            "Out-of-scope refusal must remain 1.0."
        ),
    }


def main() -> int:
    args = parse_args()
    report = asyncio.run(evaluate(args.cases.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "recommended_mode": report["recommended_mode"],
        "modes": {
            mode: {
                key: values[key]
                for key in ["recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_10", "canonical_top1_accuracy", "out_of_scope_refusal_accuracy", "latency_ms"]
            }
            for mode, values in report["modes"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
