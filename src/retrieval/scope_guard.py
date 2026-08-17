"""Deterministic scope and minimum-answerability guards for clinical queries."""

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

_DECISION_INTENT = re.compile(
    r"\b(?:refer(?:ral|red)?|investigat(?:e|ed|es|ing|ion)|recommend(?:ation|ed|s)?|"
    r"qualif(?:y|ies|ied)|eligib(?:le|ility)|scan|test(?:ed|ing)?|pathway|urgent|"
    r"offer(?:ed)?|need|should)\b",
    re.IGNORECASE,
)
_QUERY_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.IGNORECASE)
_GENERIC_DECISION_TOKENS = {
    "a", "about", "an", "and", "any", "apply", "applies", "are", "as", "at",
    "be", "been", "being", "by", "can", "cancer", "care", "clinical", "clinician",
    "could", "criteria", "criterion", "decision", "details", "determine", "determined",
    "diagnosis", "do", "does", "eligible", "eligibility", "enter", "for", "from",
    "guidance", "guideline", "have", "in", "information", "investigate", "investigation",
    "is", "it", "may", "need", "ng12", "of", "on", "or", "offered", "offer",
    "he", "her", "him", "his", "i", "me", "my", "our", "patient", "pathway",
    "person", "please", "qualify", "qualified", "qualifies", "receive", "received",
    "recommend", "recommendation", "recommended",
    "recommends", "refer", "referral", "referred", "scan", "should", "sign", "signs",
    "require", "required", "requires", "someone", "suspected", "symptom", "symptoms",
    "tell", "test", "tested", "testing", "the", "their", "they", "this", "to", "urgent",
    "us", "we", "what", "when", "whether", "who", "will", "with", "would", "you", "your",
}
_SITE_TOKENS = {
    "bladder", "bowel", "colorectal", "colon", "esophageal", "gastric", "gastrointestinal",
    "gi", "kidney", "lung", "lower", "oesophageal", "pancreatic", "pancreas", "rectal",
    "renal", "stomach", "upper", "urological",
}


def assess_query_answerability(query: str) -> dict[str, object]:
    """Fail closed only when a decision request has no patient feature at all.

    This deliberately does not encode NG12 eligibility criteria. Partially specified
    questions continue to grounded generation, which must preserve unknown qualifiers.
    """

    if not _DECISION_INTENT.search(query):
        return {"status": "model_assessed", "clinical_features": []}
    tokens = [token.lower() for token in _QUERY_TOKEN.findall(query)]
    clinical_features = sorted(
        {
            token
            for token in tokens
            if len(token) > 1
            and not token.isdigit()
            and token not in _GENERIC_DECISION_TOKENS
            and token not in _SITE_TOKENS
        }
    )
    if clinical_features:
        return {"status": "model_assessed", "clinical_features": clinical_features}
    return {
        "status": "insufficient",
        "clinical_features": [],
        "message": (
            "Insufficient information to determine whether this person meets an NG12 "
            "referral or investigation criterion. Please provide the suspected cancer "
            "site and the relevant patient details, such as age, specific symptoms or "
            "signs, their duration, smoking history, and any test results."
        ),
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
