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
_PATIENT_SPECIFIC_CONTEXT = re.compile(
    r"\b(?:(?:a|the|this|that|my|our|your)(?:\s+[a-z0-9-]+){0,3}\s+"
    r"(?:patient|person)|patient\s+(?:has|had|reports?|describes?|feels?|experiences?)|"
    r"someone|somebody|i|i'm|i've|me|my|mine|he|she|they|his|hers?|theirs?)\b",
    re.IGNORECASE,
)
_QUERY_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.IGNORECASE)
_AGE_TOKEN = re.compile(r"\d+(?:-year-old)?", re.IGNORECASE)
_NON_CLINICAL_FEATURE_TOKENS = {
    "a", "about", "adult", "age", "aged", "an", "and", "any", "apply", "applies",
    "are", "around", "as", "ask", "asked", "asking", "asks", "at",
    "be", "been", "being", "by", "can", "cancer", "care", "clinical", "clinician",
    "child", "complaint", "complaints", "concern", "concerned", "concerning", "could",
    "criteria", "criterion", "danger",
    "dangerous", "decision", "described", "describes", "describing", "details", "determine",
    "determined", "diagnosis", "do",
    "does", "eligible", "eligibility", "enter", "experience", "experienced", "experiencing",
    "feel", "feeling", "feels", "fine", "for", "from", "general", "generally", "get",
    "getting", "got", "guidance", "guideline", "had", "has", "have", "having",
    "in", "information", "investigate", "investigation", "issue", "issues",
    "is", "it", "may", "need", "ng12", "of", "on", "or", "offered", "offer",
    "day", "days", "female", "ill", "long", "male", "mean", "means", "month", "months",
    "matter", "near", "new-onset", "normal", "noticed", "noticing", "ok", "okay", "old",
    "older", "persistent",
    "he", "her", "him", "his", "i", "me", "my", "our", "patient", "pathway",
    "person", "please", "qualify", "qualified", "qualifies", "receive", "received",
    "problem", "problems", "reported", "reporting", "reports", "said", "say", "says",
    "serious", "seriously", "severity",
    "sick", "some", "somebody", "specific", "sudden",
    "someone", "something", "thing", "things", "unwell", "weird", "weirdness", "worry",
    "worried", "wrong",
    "recommend", "recommendation", "recommended",
    "recommends", "refer", "referral", "referred", "scan", "should", "sign", "signs",
    "require", "required", "requires", "someone", "suspected", "symptom", "symptoms",
    "tell", "test", "tested", "testing", "that", "the", "their", "they", "these", "this",
    "those", "to", "unexplained", "urgent",
    "us", "we", "week", "weeks", "what", "when", "whether", "who", "will", "with",
    "woman", "would", "year", "years", "you", "younger", "your",
}
_SITE_TOKENS = {
    "abdomen", "abdominal", "belly", "bladder", "bowel", "breathing", "chest", "colorectal",
    "colon", "digestive", "esophageal", "gastric", "gastrointestinal", "gi", "kidney", "lung",
    "lower", "oesophageal", "pancreatic", "pancreas", "rectal", "renal", "respiratory",
    "stomach", "tummy", "upper", "urinary", "urological",
}


def assess_query_answerability(query: str) -> dict[str, object]:
    """Fail closed when an assessment has no concrete clinical feature at all.

    This deliberately does not encode NG12 eligibility criteria or a symptom ontology.
    It removes only grammatical, workflow, site, qualifier, and explicitly vague terms.
    Partially specified questions with a concrete feature continue to grounded generation,
    which must preserve unknown qualifiers.
    """

    tokens = [token.lower() for token in _QUERY_TOKEN.findall(query)]
    clinical_features = sorted(
        {
            token
            for token in tokens
            if len(token) > 1
            and not _AGE_TOKEN.fullmatch(token)
            and token not in _NON_CLINICAL_FEATURE_TOKENS
            and token not in _SITE_TOKENS
        }
    )
    requires_clinical_feature = bool(
        _DECISION_INTENT.search(query) or _PATIENT_SPECIFIC_CONTEXT.search(query)
    )
    if clinical_features or not requires_clinical_feature:
        return {"status": "model_assessed", "clinical_features": clinical_features}
    return {
        "status": "insufficient",
        "clinical_features": [],
        "message": (
            "Insufficient information to assess this patient against NG12. Please describe "
            "the specific symptoms or signs rather than a general issue, together with "
            "relevant details such as age, duration, smoking history, and any test results."
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
