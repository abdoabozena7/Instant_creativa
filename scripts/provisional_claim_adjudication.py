"""Triage claim support with a structured Ollama judge; human review remains authoritative."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "data" / "eval"
JUDGE_MODEL = "gpt-oss:120b-cloud"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

SYSTEM_PROMPT = """You are a strict evidence adjudicator, not a medical answerer.
Evaluate only against the supplied evidence and gold behavior rubric. Do not use outside knowledge.

Rules:
1. Break the answer into atomic clinical or guideline claims. Ignore headings and purely conversational text.
2. A claim is supported only when its full meaning follows directly from supplied evidence.
3. Mark expansions, synonyms that add meaning, absolute language, invented workflow steps, and unsupported exclusions as unsupported.
4. For each claim, identify citation labels visible near that claim even if formatting contains spaces, bold markers, or unusual brackets.
5. citation_entails_claim is true only if at least one cited label directly supports the claim. A valid-looking label alone is insufficient.
6. The 2026 current source must prevail over 2015 supporting material for actions and thresholds.
7. For expected_behavior=insufficient, the answer must identify the missing qualifier or unsupported conclusion and must not invent an alternative action.
8. For expected_behavior=refuse, the configured-scope refusal must occur before clinical answering.

Return JSON only with this shape:
{
  "answer_behavior_correct": true,
  "current_guideline_accuracy": true_or_false_or_null,
  "claims": [
    {
      "claim": "atomic claim",
      "cited_labels": ["E1"],
      "supported": true,
      "citation_entails_claim": true_or_false_or_null,
      "severity": "none|minor|major",
      "reason": "short evidence-specific reason"
    }
  ],
  "failure_types": ["scope|retrieval|grounding|citation_binding|overreach|current_vs_historical|incorrect_refusal|false_refusal"],
  "summary": "short strict verdict"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packets",
        type=Path,
        default=DEFAULT_EVAL_DIR / "adjudication_packets_v1.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVAL_DIR)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_json_content(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
    return json.loads(content)


async def judge_packet(packet: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "case_id": packet["case_id"],
        "question": packet["question"],
        "expected_behavior": packet["expected_behavior"],
        "required_concepts": packet["required_concepts"],
        "gold_rationale": packet["gold_rationale"],
        "answer": packet["answer"],
        "evidence": packet["evidence"],
    }
    request = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    async with httpx.AsyncClient(timeout=240.0) as client:
        for attempt in range(2):
            response = await client.post(OLLAMA_URL, json=request)
            response.raise_for_status()
            try:
                verdict = parse_json_content(response.json()["message"]["content"])
                return {"case_id": packet["case_id"], **verdict}
            except (KeyError, json.JSONDecodeError):
                if attempt:
                    raise
    raise RuntimeError("Judge returned no parseable response")


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    packets = load_jsonl(args.packets.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "provisional_claim_adjudication_v1.jsonl"
    partial_path = output_path.with_suffix(".partial.jsonl")
    verdicts = []
    for index, packet in enumerate(packets, start=1):
        verdict = await judge_packet(packet)
        verdicts.append(verdict)
        partial_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in verdicts),
            encoding="utf-8",
        )
        unsupported = sum(not claim.get("supported", False) for claim in verdict.get("claims", []))
        print(
            f"[{index:02d}/{len(packets)}] {packet['case_id']} "
            f"behavior={verdict.get('answer_behavior_correct')} unsupported={unsupported}"
        )
    output_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in verdicts),
        encoding="utf-8",
    )
    partial_path.unlink(missing_ok=True)

    claims = [claim for verdict in verdicts for claim in verdict.get("claims", [])]
    cited_claims = [claim for claim in claims if claim.get("cited_labels")]
    supported = [bool(claim.get("supported")) for claim in claims]
    citation_entailment = [
        bool(claim.get("citation_entails_claim")) for claim in cited_claims
    ]
    behaviors = [bool(verdict.get("answer_behavior_correct")) for verdict in verdicts]
    current = [
        bool(verdict["current_guideline_accuracy"])
        for verdict in verdicts
        if verdict.get("current_guideline_accuracy") is not None
    ]
    failure_counts = Counter(
        failure for verdict in verdicts for failure in verdict.get("failure_types", [])
    )
    review_queue = [
        {
            "case_id": verdict["case_id"],
            "answer_behavior_correct": verdict.get("answer_behavior_correct"),
            "unsupported_claims": sum(
                not claim.get("supported", False) for claim in verdict.get("claims", [])
            ),
            "failure_types": verdict.get("failure_types", []),
            "summary": verdict.get("summary"),
        }
        for verdict in verdicts
        if not verdict.get("answer_behavior_correct")
        or any(not claim.get("supported", False) for claim in verdict.get("claims", []))
        or verdict.get("failure_types")
    ]
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "provisional_model_assisted_pending_human_review",
        "judge_model": JUDGE_MODEL,
        "independence_limitation": (
            "The judge is the same model family used for generation. These scores are triage signals, "
            "not independent or clinically validated semantic metrics."
        ),
        "cases": len(verdicts),
        "claims": len(claims),
        "citation_accuracy_provisional": round(sum(citation_entailment) / len(citation_entailment), 4)
        if citation_entailment
        else None,
        "claim_support_rate_provisional": round(sum(supported) / len(supported), 4)
        if supported
        else None,
        "unsupported_claim_rate_provisional": round(
            sum(not value for value in supported) / len(supported), 4
        )
        if supported
        else None,
        "answer_behavior_accuracy_provisional": round(sum(behaviors) / len(behaviors), 4),
        "current_guideline_accuracy_provisional": round(sum(current) / len(current), 4)
        if current
        else None,
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "human_review_queue": review_queue,
    }
    metrics_path = output_dir / "provisional_semantic_metrics_v1.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    report_path = output_dir / "blind_e2e_report_v1.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["semantic_metrics"]["provisional_model_assisted"] = metrics
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    metrics = asyncio.run(main_async(parse_args()))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
