"""High-precision emergency redirect before evidence retrieval."""

from __future__ import annotations

import re


EMERGENCY_REDIRECT_MESSAGE = (
    "This may need immediate medical attention. Contact your local emergency services "
    "now or go to the nearest emergency department. Do not wait for, or rely on, this "
    "evidence demo for urgent care."
)

_IMMEDIATE_CONTEXT = re.compile(
    r"\b(?:right\s+now|currently|at\s+the\s+moment|suddenly|just\s+started|"
    r"i\s+am|i['’]?m|i\s+feel|my\s+|patient\s+is|person\s+is|someone\s+is)\b",
    re.IGNORECASE,
)
_SEVERE_CONTEXT = re.compile(
    r"\b(?:severe(?:ly)?|heavy|heavily|profuse(?:ly)?|uncontrolled|cannot|can['’]?t|"
    r"unable\s+to|collapsed?|unconscious|faint(?:ed|ing)?|feel(?:ing)?\s+faint|"
    r"blue\s+lips?|choking|not\s+breathing|struggling\s+to\s+breathe)\b",
    re.IGNORECASE,
)
_EMERGENCY_FEATURES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "major_bleeding",
        re.compile(
            r"\b(?:(?:very\s+)?(?:heavy|profuse|uncontrolled)\s+bleed(?:ing)?|"
            r"vomit(?:ing|ed)?\s+(?:a\s+lot\s+of\s+)?blood|haematemesis|"
            r"hematemesis|cough(?:ing|ed)?\s+(?:up\s+)?blood|bleed(?:ing)?\s+(?:a\s+lot|heavily|profusely))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "breathing_emergency",
        re.compile(
            r"\b(?:cannot|can['’]?t|unable\s+to|struggling\s+to)\s+(?:breathe|breath)\b|"
            r"\b(?:not\s+breathing|blue\s+lips?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "loss_of_consciousness",
        re.compile(
            r"\b(?:collapsed?|unconscious|passed\s+out|fainted|fainting)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "acute_chest_pain",
        re.compile(
            r"\b(?:severe|crushing|sudden)\s+chest\s+pain\b|"
            r"\bchest\s+pain\b.{0,45}\b(?:faint|sweat|breath)\w*\b",
            re.IGNORECASE,
        ),
    ),
)


def assess_emergency(query: str) -> dict[str, object]:
    """Redirect only explicit, current, high-acuity scenarios.

    General questions about emergency symptoms remain answerable. A match requires an
    emergency feature plus immediate/patient context and, except for intrinsically acute
    features, a severity signal. This intentionally favors precision over recall.
    """

    immediate = bool(_IMMEDIATE_CONTEXT.search(query))
    severe = bool(_SEVERE_CONTEXT.search(query))
    reasons = [name for name, pattern in _EMERGENCY_FEATURES if pattern.search(query)]
    intrinsically_acute = any(
        reason in {"breathing_emergency", "loss_of_consciousness", "acute_chest_pain"}
        for reason in reasons
    )
    if reasons and immediate and (severe or intrinsically_acute):
        return {
            "status": "redirect",
            "reason_codes": reasons,
            "message": EMERGENCY_REDIRECT_MESSAGE,
        }
    return {"status": "clear", "reason_codes": [], "message": None}
