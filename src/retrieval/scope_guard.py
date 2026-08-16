"""Deterministic clinical-scope guard applied before retrieval or generation."""

from __future__ import annotations

import re


SELECTED_PATTERNS = {
    "lung": r"\blung\b",
    "colorectal": r"\bcolorectal\b|\bbowel cancer\b|\bcolon cancer\b|\brectal cancer\b",
    "oesophageal": r"\boesophag(?:eal|us)\b|\besophag(?:eal|us)\b",
    "stomach": r"\bstomach\b|\bgastric\b",
    "pancreatic": r"\bpancrea(?:s|tic)\b",
    "bladder": r"\bbladder\b",
    "renal": r"\brenal\b|\bkidney cancer\b",
}

EXCLUDED_PATTERNS = {
    "mesothelioma": r"\bmesothelioma\b",
    "anal cancer": r"\banal cancer\b",
    "gall bladder cancer": r"\bgall\s*bladder\b|\bgallbladder\b",
    "liver cancer": r"\bliver cancer\b|\bhepatocellular\b",
    "prostate cancer": r"\bprostate\b|\bpsa\b",
    "testicular cancer": r"\btesticular\b|\btesticle cancer\b",
    "penile cancer": r"\bpenile\b|\bpenis cancer\b",
    "breast cancer": r"\bbreast\b",
    "gynaecological cancer": r"\bovarian\b|\bcervical\b|\bendometrial\b|\bvulval\b|\bvaginal cancer\b",
    "skin cancer": r"\bmelanoma\b|\bskin cancer\b",
    "haematological cancer": r"\bleuka?emia\b|\blymphoma\b|\bmyeloma\b",
}


def assess_scope(query: str) -> dict[str, object]:
    excluded = [
        site for site, pattern in EXCLUDED_PATTERNS.items() if re.search(pattern, query, re.I)
    ]
    # Remove excluded phrases before looking for selected sites. This prevents
    # an excluded phrase such as "gall bladder" from also satisfying the
    # independent in-scope word-boundary match for "bladder". A genuinely
    # mixed query still retains any separately named in-scope site.
    selected_text = query
    for pattern in EXCLUDED_PATTERNS.values():
        selected_text = re.sub(pattern, " ", selected_text, flags=re.I)
    selected = [
        site
        for site, pattern in SELECTED_PATTERNS.items()
        if re.search(pattern, selected_text, re.I)
    ]
    if excluded and not selected:
        return {
            "status": "out_of_scope",
            "selected_sites": [],
            "excluded_sites": excluded,
            "message": (
                "This NG12 demo is restricted to lung, colorectal, oesophageal, stomach, "
                "pancreatic, bladder, and renal cancer. The requested cancer site is outside "
                "the configured evidence scope."
            ),
        }
    return {
        "status": "in_scope",
        "selected_sites": selected,
        "excluded_sites": excluded,
        "message": (
            "The query mentions an excluded site as well as an in-scope site; only in-scope "
            "evidence will be searched."
            if excluded
            else None
        ),
    }
