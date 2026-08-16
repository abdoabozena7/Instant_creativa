"""Prepare, execute, resume, and score the NG12 automated multi-judge evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_blind_e2e_evaluation import verify_freeze  # noqa: E402
from src.evaluation.judge_clients import (  # noqa: E402
    GeminiJudge,
    OllamaJudge,
    SemanticJudge,
)
from src.evaluation.judge_prompts import prompt_and_schema  # noqa: E402
from src.evaluation.multi_judge import (  # noqa: E402
    build_tasks,
    load_jsonl,
    score_consensus,
    score_hackathon,
    validate_judgment,
    write_jsonl,
)


EVAL_DIR = PROJECT_ROOT / "data" / "eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument(
        "--mode",
        choices=("hackathon", "full"),
        default="hackathon",
        help="Hackathon mode uses one primary pass plus conditional escalation; full runs 2x3.",
    )
    parser.add_argument("--passes", type=int, default=3, choices=[3])
    parser.add_argument("--concurrency", type=int, default=3, choices=range(1, 9))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call both judge APIs. Without this flag only tasks are prepared.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Diagnostic limit over judge/pass items. Limited runs cannot be scored.",
    )
    return parser.parse_args()


def prepare(eval_dir: Path, mode: str = "hackathon") -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict]:
    freeze = verify_freeze(eval_dir / "evaluation_freeze.json")
    packets = load_jsonl(eval_dir / "adjudication_packets_v1.jsonl")
    provisional = load_jsonl(eval_dir / "provisional_claim_adjudication_v1.jsonl")
    tasks, deterministic = build_tasks(packets, provisional)
    write_jsonl(eval_dir / "multi_judge_tasks_v1.jsonl", tasks)
    write_jsonl(
        eval_dir / "multi_judge_deterministic_v1.jsonl", deterministic
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture_sha256": freeze["architecture_sha256"],
        "llm_tasks": len(tasks),
        "deterministic_missing_citation_tasks": len(deterministic),
        "mode": mode,
        "passes": 1 if mode == "hackathon" else 3,
        "judges": {
            "gemini": "gemini-3.6-flash",
            "gpt_oss": "gpt-oss:120b-cloud",
        },
        "planned_model_calls": (
            {
                "primary": len(tasks),
                "conditional_second_judge": f"0-{len(tasks)}",
                "maximum": len(tasks) * 2,
            }
            if mode == "hackathon"
            else len(tasks) * 3 * 2
        ),
        "policy": (
            "One Gemini pass per task; only primary failures are escalated to GPT-OSS. "
            "Escalation disagreement is UNCERTAIN and fails closed. Automated evaluation, "
            "not clinical validation."
            if mode == "hackathon"
            else "Each task evaluates exactly one semantic criterion. Three passes per judge; "
            "cross-judge disagreement is UNCERTAIN and fails closed. Automated evaluation, "
            "not clinical validation."
        ),
    }
    (eval_dir / "multi_judge_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return tasks, deterministic, {"freeze": freeze, "manifest": manifest}


def execution_key(task_id: str, judge: str, pass_number: int) -> tuple[str, str, int]:
    return task_id, judge, pass_number


async def call_with_retry(
    judge: SemanticJudge,
    task: dict[str, Any],
    pass_number: int,
    *,
    attempts: int = 3,
) -> dict[str, Any]:
    prompt, schema = prompt_and_schema(task)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            judgment = await judge.judge(prompt, schema)
            validate_judgment(task, judgment)
            return {
                "task_id": task["task_id"],
                "input_sha256": task["input_sha256"],
                "criterion": task["criterion"],
                "case_id": task["case_id"],
                "claim_id": task.get("claim_id"),
                "judge": judge.name,
                "model": judge.model,
                "pass": pass_number,
                "label": judgment["label"],
                "reason": judgment["reason"],
                "claim_ids": judgment.get("claim_ids", []),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "attempt": attempt,
            }
        except (httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < attempts:
                await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"{task['task_id']} / {judge.name} / pass {pass_number} failed after {attempts} attempts: {last_error}"
    )


async def execute(
    tasks: list[dict[str, Any]],
    *,
    passes: int,
    concurrency: int,
    output_path: Path,
    max_items: int | None,
    judges: list[SemanticJudge] | None = None,
) -> list[dict[str, Any]]:
    judges = judges or [GeminiJudge(), OllamaJudge()]
    existing = load_jsonl(output_path) if output_path.is_file() else []
    completed = {
        execution_key(row["task_id"], row["judge"], row["pass"])
        for row in existing
    }
    items = [
        (task, judge, pass_number)
        for task in tasks
        for judge in judges
        for pass_number in range(1, passes + 1)
        if execution_key(task["task_id"], judge.name, pass_number) not in completed
    ]
    random.Random(12).shuffle(items)
    if max_items is not None:
        items = items[:max_items]
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    rows = list(existing)
    total = len(items)

    async def run_one(
        index: int,
        task: dict[str, Any],
        judge: SemanticJudge,
        pass_number: int,
    ) -> None:
        async with semaphore:
            result = await call_with_retry(judge, task, pass_number)
        async with write_lock:
            rows.append(result)
            with output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"[{index:04d}/{total:04d}] {task['task_id']} "
                f"judge={judge.name} pass={pass_number} label={result['label']}"
            )

    await asyncio.gather(
        *(
            run_one(index, task, judge, pass_number)
            for index, (task, judge, pass_number) in enumerate(items, start=1)
        )
    )
    return rows


async def async_main(args: argparse.Namespace) -> int:
    eval_dir = args.eval_dir.resolve()
    tasks, deterministic, context = prepare(eval_dir, args.mode)
    manifest = context["manifest"]
    print(json.dumps(manifest, indent=2))
    if not args.execute:
        print(
            "Prepared only. Set GEMINI_API_KEY in the environment, then rerun with --execute."
        )
        return 0

    judgment_path = eval_dir / f"{args.mode}_judge_raw_v1.jsonl"
    if args.mode == "hackathon":
        judgments = await execute(
            tasks,
            passes=1,
            concurrency=args.concurrency,
            output_path=judgment_path,
            max_items=args.max_items,
            judges=[GeminiJudge()],
        )
        if args.max_items is None:
            primary_by_task = {
                row["task_id"]: row
                for row in judgments
                if row["judge"] == "gemini"
            }
            pass_labels = {
                "claim_support": "SUPPORTED",
                "citation_entailment": "YES",
                "overreach": "NO",
                "completeness": "YES",
                "refusal_quality": "YES",
            }
            escalations = [
                task
                for task in tasks
                if primary_by_task[task["task_id"]]["label"]
                != pass_labels[task["criterion"]]
            ]
            print(f"Escalating {len(escalations)} primary failures to GPT-OSS.")
            judgments = await execute(
                escalations,
                passes=1,
                concurrency=args.concurrency,
                output_path=judgment_path,
                max_items=None,
                judges=[OllamaJudge()],
            )
    else:
        judgments = await execute(
            tasks,
            passes=args.passes,
            concurrency=args.concurrency,
            output_path=judgment_path,
            max_items=args.max_items,
        )
    if args.max_items is not None:
        print("Diagnostic limited run complete; consensus scoring intentionally skipped.")
        return 0

    if args.mode == "hackathon":
        report, consensus = score_hackathon(
            tasks,
            deterministic,
            judgments,
            models=manifest["judges"],
            architecture_sha256=context["freeze"]["architecture_sha256"],
        )
    else:
        report, consensus = score_consensus(
            tasks,
            deterministic,
            judgments,
            passes=args.passes,
            judge_names=("gemini", "gpt_oss"),
            models=manifest["judges"],
            architecture_sha256=context["freeze"]["architecture_sha256"],
        )
    write_jsonl(eval_dir / "multi_judge_consensus_v1.jsonl", consensus)
    (eval_dir / "multi_judge_report_v1.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
