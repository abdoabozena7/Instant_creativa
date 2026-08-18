"""Run one retrieval-only baseline/reranker comparison on frozen blind questions."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_retrieval_noise import load_jsonl, relevant  # noqa: E402
from src.retrieval.engine import RetrievalEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "blind_questions_v1.jsonl",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "blind_gold_v1.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "eval"
            / "blind_retrieval_rerank_experiment_v13.json"
        ),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def evaluate_configuration(
    engine: RetrievalEngine,
    questions: list[dict[str, Any]],
    gold_by_id: dict[str, dict[str, Any]],
    *,
    rerank: bool,
) -> dict[str, Any]:
    ranks: list[int | None] = []
    gold_at_3: list[int] = []
    gold_at_5: list[int] = []
    latencies: list[float] = []
    cases: list[dict[str, Any]] = []

    for question in questions:
        gold = gold_by_id[question["case_id"]]
        if not (
            gold.get("acceptable_recommendation_ids")
            or gold.get("acceptable_content_types")
            or gold.get("acceptable_record_ids")
        ):
            continue
        response = await engine.search(
            question["question"],
            mode="hybrid",
            top_k=5,
            rerank=rerank,
        )
        results = response["results"]
        rank = next(
            (
                index
                for index, result in enumerate(results, start=1)
                if relevant(result, gold)
            ),
            None,
        )
        hits_at_3 = sum(relevant(result, gold) for result in results[:3])
        hits_at_5 = sum(relevant(result, gold) for result in results[:5])
        ranks.append(rank)
        gold_at_3.append(hits_at_3)
        gold_at_5.append(hits_at_5)
        latencies.append(float(response["latency_ms"]))
        cases.append(
            {
                "case_id": question["case_id"],
                "rank": rank,
                "gold_results_at_3": hits_at_3,
                "gold_results_at_5": hits_at_5,
                "top_3": [
                    {
                        "rank": result["rank"],
                        "chunk_id": result["chunk_id"],
                        "recommendation_id": result.get("recommendation_id"),
                        "content_type": result["content_type"],
                        "cancer_sites": result.get("cancer_sites", []),
                        "is_gold": relevant(result, gold),
                    }
                    for result in results[:3]
                ],
            }
        )

    count = len(ranks)
    return {
        "reranker_enabled": rerank,
        "queries_scored": count,
        "recall_at_1": round(sum(rank == 1 for rank in ranks) / count, 4),
        "recall_at_3": round(
            sum(rank is not None and rank <= 3 for rank in ranks) / count, 4
        ),
        "recall_at_5": round(
            sum(rank is not None and rank <= 5 for rank in ranks) / count, 4
        ),
        "precision_at_3": round(sum(gold_at_3) / (count * 3), 4),
        "precision_at_5": round(sum(gold_at_5) / (count * 5), 4),
        "mrr_at_5": round(sum(1 / rank if rank else 0 for rank in ranks) / count, 4),
        "latency_ms": {
            "p50": round(statistics.median(latencies), 2),
            "maximum": round(max(latencies), 2),
        },
        "cases": cases,
    }


async def evaluate(questions_path: Path, gold_path: Path) -> dict[str, Any]:
    questions = load_jsonl(questions_path)
    gold_rows = load_jsonl(gold_path)
    gold_by_id = {row["case_id"]: row for row in gold_rows}
    engine = RetrievalEngine()
    baseline = await evaluate_configuration(
        engine, questions, gold_by_id, rerank=False
    )
    reranked = await evaluate_configuration(engine, questions, gold_by_id, rerank=True)

    scored_gold = [
        gold
        for gold in gold_rows
        if gold.get("acceptable_recommendation_ids")
        or gold.get("acceptable_content_types")
        or gold.get("acceptable_record_ids")
    ]
    capacity_at_3 = sum(
        min(sum(relevant(chunk, gold) for chunk in engine.chunks), 3)
        for gold in scored_gold
    )
    denominator = len(scored_gold) * 3
    return {
        "document": "NICE NG12",
        "evaluation_name": "blind_retrieval_rerank_experiment_v13",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_policy": {
            "configuration_selected_on": "data/eval/retrieval_cases.jsonl",
            "blind_run_policy": (
                "One retrieval-only comparison after selecting the reranker on the "
                "development set; no parameter changes are made from this result."
            ),
            "frozen_v12_use": (
                "The completed v12 run was inspected retrospectively only to diagnose "
                "the strict label ceiling and non-gold categories."
            ),
        },
        "architecture_files_sha256": {
            "src/retrieval/engine.py": file_sha256(
                PROJECT_ROOT / "src" / "retrieval" / "engine.py"
            ),
            "src/retrieval/reranker.py": file_sha256(
                PROJECT_ROOT / "src" / "retrieval" / "reranker.py"
            ),
        },
        "strict_precision_at_3_ceiling": round(capacity_at_3 / denominator, 4),
        "baseline": baseline,
        "hybrid_reranked": reranked,
        "delta": {
            "precision_at_3": round(
                reranked["precision_at_3"] - baseline["precision_at_3"], 4
            ),
            "recall_at_3": round(
                reranked["recall_at_3"] - baseline["recall_at_3"], 4
            ),
            "recall_at_1": round(
                reranked["recall_at_1"] - baseline["recall_at_1"], 4
            ),
            "mrr_at_5": round(reranked["mrr_at_5"] - baseline["mrr_at_5"], 4),
        },
    }


def main() -> int:
    args = parse_args()
    report = asyncio.run(evaluate(args.questions.resolve(), args.gold.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "strict_precision_at_3_ceiling": report[
                    "strict_precision_at_3_ceiling"
                ],
                "baseline": {
                    key: report["baseline"][key]
                    for key in ["precision_at_3", "recall_at_1", "recall_at_3", "mrr_at_5"]
                },
                "hybrid_reranked": {
                    key: report["hybrid_reranked"][key]
                    for key in ["precision_at_3", "recall_at_1", "recall_at_3", "mrr_at_5"]
                },
                "delta": report["delta"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
