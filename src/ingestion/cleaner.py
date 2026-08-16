"""Conservative cleanup for native NG12 PDF text.

The functions here remove layout noise and repair line wrapping. They do not
summarise, paraphrase, or derive clinical facts.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


REPEATED_HEADER = "Suspected cancer: recognition and referral (NG12)"
FOOTER_START = re.compile(r"^©\s+NICE\s+2026\.")
FOOTER_CONTINUATION = re.compile(r"conditions#notice-of-rights\)")
WHITESPACE = re.compile(r"[ \t]+")
RECOMMENDATION_ID = re.compile(r"^1\.\d+(?:\.\d+)+\b")


def clean_page_text(raw_text: str, page_number: int) -> tuple[str, dict[str, int]]:
    """Remove repeat furniture while retaining the source's line structure."""

    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned: list[str] = []
    removed_headers = 0
    removed_footer_lines = 0
    in_footer = False

    for line in lines:
        stripped = WHITESPACE.sub(" ", line.strip())

        if page_number > 1 and stripped == REPEATED_HEADER:
            removed_headers += 1
            continue

        if FOOTER_START.match(stripped):
            in_footer = True
            removed_footer_lines += 1
            continue
        if in_footer:
            if stripped:
                removed_footer_lines += 1
            continue
        if FOOTER_CONTINUATION.search(stripped):
            removed_footer_lines += 1
            continue

        # PyMuPDF exposes the indented secondary bullet as U+FF0D. Normalise
        # the marker only; the NICE wording following it is untouched.
        if stripped.startswith("－"):
            stripped = "-" + stripped[1:]

        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        cleaned.append(stripped)

    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    return "\n".join(cleaned), {
        "headers_removed": removed_headers,
        "footer_lines_removed": removed_footer_lines,
    }


def join_wrapped_lines(lines: Iterable[str]) -> str:
    """Join visual line wraps while preserving list-item boundaries."""

    paragraphs: list[str] = []
    current = ""

    for raw_line in lines:
        line = WHITESPACE.sub(" ", raw_line.strip())
        if not line:
            if current:
                paragraphs.append(current)
                current = ""
            continue

        is_bullet = line.startswith(("•", "-"))
        if is_bullet:
            if current:
                paragraphs.append(current)
            current = line
            continue

        if not current:
            current = line
        elif current.endswith("-"):
            current += line
        else:
            current += " " + line

    if current:
        paragraphs.append(current)
    return "\n".join(paragraphs)


def normalise_cell_text(lines: Iterable[str]) -> str:
    """Join words recovered geometrically from a single table cell."""

    return join_wrapped_lines(lines)
