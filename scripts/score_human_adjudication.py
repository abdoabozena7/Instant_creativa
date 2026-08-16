"""Score finalized case- and claim-level human adjudication CSV exports."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
TRUE_VALUES = {"true", "1", "yes"}
FALSE_VALUES = {"false", "0", "no"}
NA_VALUES = {"not_applicable", "not applicable", "n/a", "na"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-review",
        type=Path,
        default=EVAL_DIR / "human_case_review_v1.csv",
        help="CSV export of the completed Case Review worksheet.",
    )
    parser.add_argument(
        "--claim-review",
        type=Path,
        default=EVAL_DIR / "human_claim_review_v1.csv",
        help="CSV export of the completed Claim Review worksheet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_DIR / "human_adjudication_metrics_v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=EVAL_DIR / "blind_e2e_report_v1.json",
        help="Blind report to update after successful complete adjudication.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def normalized(row: dict[str, str], field: str, row_id: str) -> str:
    if field not in row:
        raise ValueError(f"{row_id or 'unidentified row'}: missing column {field!r}")
    value = (row[field] or "").strip().lower()
    if not value or value == "uncertain":
        raise ValueError(f"{row_id or 'unidentified row'}: unresolved {field}")
    return value


def decision(
    row: dict[str, str],
    field: str,
    row_id: str,
    *,
    allow_na: bool = False,
) -> bool | None:
    value = normalized(row, field, row_id)
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    if allow_na and value in NA_VALUES:
        return None
    suffix = "true/false/not_applicable" if allow_na else "true/false"
    raise ValueError(f"{row_id}: {field} must be {suffix}")


def rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def score_rows(
    cases: list[dict[str, str]], claims: list[dict[str, str]]
) -> dict[str, Any]:
    if len(cases) != 44:
        raise ValueError(f"Expected 44 case rows, found {len(cases)}")
    if not claims:
        raise ValueError("Claim review CSV is empty")

    case_ids = [row.get("Case ID", "").strip() for row in cases]
    if any(not value for value in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("Case IDs must be present and unique")

    behavior: list[bool] = []
    current: list[bool] = []
    failure_counts: Counter[str] = Counter()
    for row, case_id in zip(cases, case_ids, strict=True):
        behavior_value = decision(row, "Final behavior correct", case_id)
        assert behavior_value is not None
        behavior.append(behavior_value)
        completeness = decision(row, "Final claim list complete", case_id)
        if completeness is not True:
            raise ValueError(
                f"{case_id}: claim list is incomplete; official claim metrics are blocked"
            )
        current_value = decision(
            row, "Final current accuracy", case_id, allow_na=True
        )
        if current_value is not None:
            current.append(current_value)
        for item in re.split(r"[,;|]", row.get("Final failure types", "")):
            item = item.strip().lower().replace(" ", "_")
            if item:
                failure_counts[item] += 1

    claim_keys = [row.get("Claim key", "").strip() for row in claims]
    if any(not value for value in claim_keys) or len(set(claim_keys)) != len(claim_keys):
        raise ValueError("Claim keys must be present and unique")

    supported: list[bool] = []
    citation_entailment: list[bool] = []
    severity_counts: Counter[str] = Counter()
    for row, claim_key in zip(claims, claim_keys, strict=True):
        support_value = decision(
            row, "Final supported", claim_key, allow_na=True
        )
        if support_value is not None:
            supported.append(support_value)
        cited = bool(row.get("Cited labels", "").strip())
        entailment = decision(
            row, "Final citation entails", claim_key, allow_na=not cited
        )
        if cited:
            if entailment is None:
                raise ValueError(
                    f"{claim_key}: cited claim cannot use NOT_APPLICABLE for entailment"
                )
            citation_entailment.append(entailment)
        severity = normalized(row, "Final severity", claim_key)
        if severity not in {"none", "minor", "major", *NA_VALUES}:
            raise ValueError(f"{claim_key}: invalid Final severity {severity!r}")
        if severity not in NA_VALUES:
            severity_counts[severity] += 1

    unsupported = [not value for value in supported]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_human_adjudication",
        "cases_adjudicated": len(cases),
        "claims_proposed": len(claims),
        "claims_scored": len(supported),
        "citation_accuracy": rate(citation_entailment),
        "claim_support_rate": rate(supported),
        "unsupported_claim_rate": rate(unsupported),
        "answer_behavior_accuracy": rate(behavior),
        "current_guideline_accuracy_human": rate(current),
        "counts": {
            "supported_claims": sum(supported),
            "unsupported_claims": sum(unsupported),
            "cited_claims_checked": len(citation_entailment),
            "citations_entailed": sum(citation_entailment),
            "current_guideline_cases_checked": len(current),
        },
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "method": (
            "Two independent reviewer columns followed by adjudicated Final columns. "
            "Only Final columns are scored; incomplete or uncertain decisions fail closed."
        ),
    }


def main() -> int:
    args = parse_args()
    metrics = score_rows(
        load_csv(args.case_review.resolve()),
        load_csv(args.claim_review.resolve()),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["semantic_metrics"] = {
        "status": "complete_human_adjudication",
        "citation_accuracy": metrics["citation_accuracy"],
        "claim_support_rate": metrics["claim_support_rate"],
        "unsupported_claim_rate": metrics["unsupported_claim_rate"],
        "answer_behavior_accuracy": metrics["answer_behavior_accuracy"],
        "current_guideline_accuracy_human": metrics[
            "current_guideline_accuracy_human"
        ],
        "human_adjudication": metrics,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
