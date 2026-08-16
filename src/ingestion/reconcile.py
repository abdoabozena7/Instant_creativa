"""Reconcile current and historical NG12 records without merging clinical wording."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any

from .models import NormalizedRecord


TOKEN = re.compile(r"[a-z]+|\d+(?:\.\d+)?")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
RECOMMENDATION_ID = re.compile(r"\b1\.\d+(?:\.\d+)+\b")
WITHIN_TWO_WEEKS = re.compile(r"\(?for an appointment within 2 weeks\)?|within 2 weeks", re.I)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "if",
    "in",
    "is",
    "of",
    "or",
    "people",
    "the",
    "they",
    "to",
    "with",
}
ACTION_PATTERNS = {
    "suspected_cancer_pathway": re.compile(r"suspected cancer pathway", re.I),
    "direct_access": re.compile(r"direct access", re.I),
    "fit": re.compile(r"faecal immunochemical|\bFIT\b", re.I),
    "occult_blood": re.compile(r"occult blood", re.I),
    "chest_xray": re.compile(r"chest X-ray", re.I),
    "ct": re.compile(r"\bCT\b", re.I),
    "ultrasound": re.compile(r"ultrasound", re.I),
    "endoscopy": re.compile(r"endoscopy", re.I),
    "non_urgent": re.compile(r"non[-–]urgent", re.I),
}

# These sections preserve the same recommendation order between the full 2015
# guideline and the current numbered guideline. Explicit structural mapping is
# safer than lexical similarity for short recommendations sharing an investigation.
STRUCTURAL_ID_MAP = {
    "Lung cancer": ["1.1.1", "1.1.2", "1.1.3"],
    "Oesophageal cancer": ["1.2.1", "1.2.2", "1.2.3"],
    "Pancreatic cancer": ["1.2.4", "1.2.5"],
    "Stomach cancer": ["1.2.6", "1.2.7", "1.2.8", "1.2.9"],
    "Bladder cancer": ["1.6.4", "1.6.5"],
    "Renal cancer": ["1.6.6"],
    "Non-site-specific symptoms": ["1.13.2", "1.13.3", "1.13.4"],
}


def reconcile_records(
    current_records: list[NormalizedRecord], full_records: list[NormalizedRecord]
) -> tuple[list[NormalizedRecord], dict[str, Any]]:
    current_recommendations = [
        record
        for record in current_records
        if record.recommendation_id is not None and record.canonical_recommendation
    ]
    current_by_id = {
        record.recommendation_id: record for record in current_recommendations
    }
    current_by_site: dict[str, list[NormalizedRecord]] = defaultdict(list)
    for record in current_recommendations:
        for site in record.cancer_sites:
            current_by_site[site].append(record)

    duplicates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    unmatched_historical: list[str] = []

    for record in full_records:
        if record.content_type == "historical_recommendation":
            candidates = _candidate_current_records(record, current_recommendations, current_by_site)
            match = _structural_match(record, current_by_id)
            structurally_matched = match is not None
            if match is not None:
                score = _similarity(record.text, match.text)
            else:
                match, score = _best_match(record.text, candidates)
            if match is None or score < 0.32:
                unmatched_historical.append(record.record_id)
                continue
            record.related_recommendation_ids = [str(match.recommendation_id)]
            match.supporting_record_ids.append(record.record_id)

            if score >= 0.62:
                duplicates.append(
                    {
                        "supporting_record_id": record.record_id,
                        "canonical_record_id": match.record_id,
                        "recommendation_id": match.recommendation_id,
                        "similarity": round(score, 4),
                        "resolution": "2026 current recommendation retained; historical copy is not retrieval eligible.",
                    }
                )

            reasons = _possible_conflict_reasons(record.text, match.text)
            if reasons and (structurally_matched or score >= 0.42):
                record.conflict_status = "possible_version_difference"
                conflicts.append(
                    {
                        "supporting_record_id": record.record_id,
                        "canonical_record_id": match.record_id,
                        "recommendation_id": match.recommendation_id,
                        "similarity": round(score, 4),
                        "reasons": reasons,
                        "resolution": "Not auto-resolved; 2026 wording remains canonical and the difference is flagged for review.",
                    }
                )
            continue

        related_ids = _broad_support_links(record, current_recommendations, current_by_site)
        record.related_recommendation_ids = related_ids
        for recommendation_id in related_ids:
            current_by_id[recommendation_id].supporting_record_ids.append(record.record_id)

    for record in current_records:
        record.supporting_record_ids = list(dict.fromkeys(record.supporting_record_ids))

    merged = current_records + full_records
    merged.sort(
        key=lambda record: (
            0 if record.source_type == "current_guideline" else 1,
            record.page,
            record.record_id,
        )
    )
    diagnostics = {
        "records_from_current_pdf": len(current_records),
        "records_from_full_pdf": len(full_records),
        "records_retained_after_scope_filtering": len(merged),
        "duplicates_detected": duplicates,
        "duplicates_detected_count": len(duplicates),
        "possible_conflicts": conflicts,
        "conflicts_detected_count": len(conflicts),
        "unmatched_historical_recommendations": unmatched_historical,
        "records_by_source_and_content_type": {
            source: dict(sorted(counts.items()))
            for source, counts in _counts_by_source(merged).items()
        },
    }
    return merged, diagnostics


def _candidate_current_records(
    record: NormalizedRecord,
    all_current: list[NormalizedRecord],
    by_site: dict[str, list[NormalizedRecord]],
) -> list[NormalizedRecord]:
    candidates: list[NormalizedRecord] = []
    for site in record.cancer_sites:
        candidates.extend(by_site.get(site, []))
    if record.section == "Patient information and support":
        candidates.extend(item for item in all_current if str(item.recommendation_id).startswith("1.14."))
    elif record.section == "Safety netting":
        candidates.extend(item for item in all_current if str(item.recommendation_id).startswith("1.15."))
    elif record.section == "The diagnostic process":
        candidates.extend(item for item in all_current if str(item.recommendation_id).startswith("1.16."))
    elif record.section == "Non-site-specific symptoms":
        candidates.extend(item for item in all_current if str(item.recommendation_id).startswith("1.13."))
    return list({item.record_id: item for item in candidates}.values())


def _structural_match(
    record: NormalizedRecord,
    current_by_id: dict[str, NormalizedRecord],
) -> NormalizedRecord | None:
    ordered_ids = STRUCTURAL_ID_MAP.get(record.section)
    ordinal = record.metadata.get("historical_ordinal")
    if not ordered_ids or not isinstance(ordinal, int) or not 1 <= ordinal <= len(ordered_ids):
        return None
    return current_by_id.get(ordered_ids[ordinal - 1])


def _broad_support_links(
    record: NormalizedRecord,
    current: list[NormalizedRecord],
    by_site: dict[str, list[NormalizedRecord]],
) -> list[str]:
    linked: list[NormalizedRecord] = []
    for site in record.cancer_sites:
        linked.extend(by_site.get(site, []))
    prefix = None
    if record.section == "Patient information and support":
        prefix = "1.14."
    elif record.section == "Safety netting":
        prefix = "1.15."
    elif record.section == "The diagnostic process":
        prefix = "1.16."
    elif record.section == "Non-site-specific symptoms":
        prefix = "1.13."
    if prefix:
        linked.extend(item for item in current if str(item.recommendation_id).startswith(prefix))
    return sorted(
        {
            str(item.recommendation_id)
            for item in linked
            if item.recommendation_id is not None
        },
        key=lambda value: tuple(int(part) for part in value.split(".")),
    )


def _best_match(
    historical_text: str, candidates: list[NormalizedRecord]
) -> tuple[NormalizedRecord | None, float]:
    best: NormalizedRecord | None = None
    best_score = 0.0
    for candidate in candidates:
        score = _similarity(historical_text, candidate.text)
        if score > best_score:
            best = candidate
            best_score = score
    return best, best_score


def _similarity(left: str, right: str) -> float:
    left_normal = _normalise(left)
    right_normal = _normalise(right)
    left_tokens = set(TOKEN.findall(left_normal)) - STOPWORDS
    right_tokens = set(TOKEN.findall(right_normal)) - STOPWORDS
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, left_normal, right_normal).ratio()
    return 0.6 * jaccard + 0.4 * sequence


def _normalise(text: str) -> str:
    value = text.lower().replace("–", "-").replace("‑", "-")
    value = WITHIN_TWO_WEEKS.sub("", value)
    value = YEAR.sub("", value)
    value = RECOMMENDATION_ID.sub("", value)
    return " ".join(TOKEN.findall(value))


def _possible_conflict_reasons(historical: str, current: str) -> list[str]:
    reasons: list[str] = []
    old_numbers = _clinical_numbers(historical)
    new_numbers = _clinical_numbers(current)
    if old_numbers != new_numbers and (old_numbers or new_numbers):
        reasons.append(
            f"numeric/threshold tokens differ (2015={sorted(old_numbers)}, 2026={sorted(new_numbers)})"
        )
    old_actions = {name for name, pattern in ACTION_PATTERNS.items() if pattern.search(historical)}
    new_actions = {name for name, pattern in ACTION_PATTERNS.items() if pattern.search(current)}
    if old_actions and new_actions and old_actions != new_actions:
        reasons.append(
            f"action/investigation wording differs (2015={sorted(old_actions)}, 2026={sorted(new_actions)})"
        )
    return reasons


def _clinical_numbers(text: str) -> set[str]:
    value = WITHIN_TWO_WEEKS.sub("", text)
    value = YEAR.sub("", value)
    value = RECOMMENDATION_ID.sub("", value)
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", value))


def _counts_by_source(
    records: list[NormalizedRecord],
) -> dict[str, Counter[str]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        result[record.source_version][record.content_type] += 1
    return result
