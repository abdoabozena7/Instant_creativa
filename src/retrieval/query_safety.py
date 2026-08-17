"""Deterministic guard for explicit control-plane instructions in user queries.

This is intentionally not a general-purpose prompt-injection classifier. It blocks
clear attempts to override the instruction hierarchy, bypass supplied evidence,
fabricate provenance, or extract runtime secrets before retrieval or generation.
"""

from __future__ import annotations

import re
import unicodedata


SAFETY_REFUSAL_MESSAGE = (
    "This request contains instructions that attempt to override the evidence-only "
    "workflow or manipulate its provenance. Ask a clinical NG12 question without "
    "instructions about system prompts, hidden configuration, fabricated evidence, "
    "or changing the authority of the supplied sources."
)


def _pattern(expression: str) -> re.Pattern[str]:
    return re.compile(expression, re.IGNORECASE | re.DOTALL)


_CONTROL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        _pattern(
            r"\b(?:ignore|disregard|forget|bypass|override|supersede|replace|"
            r"do\s+not\s+(?:follow|obey)|don't\s+(?:follow|obey)|stop\s+following)\b"
            r".{0,90}\b(?:previous|prior|earlier|above|system|developer|hidden|original|"
            r"instructions?|rules?|prompt|policy|guardrails?)\b"
        ),
    ),
    (
        "role_override",
        _pattern(
            r"\b(?:enable|enter|activate|switch\s+to)\b.{0,35}"
            r"\b(?:developer|jailbreak|dan|unrestricted)\s+mode\b"
        ),
    ),
    (
        "role_override",
        _pattern(
            r"(?:^|\n|\s)(?:#{1,6}\s*|[<\[]\s*)?(?:system|developer)\s*"
            r"(?:override|message|instruction|prompt)\b"
        ),
    ),
    (
        "evidence_bypass",
        _pattern(
            r"\b(?:(?:do\s+not|don't|never)\s+(?:use|cite|follow|obey)|"
            r"(?:ignore|discard)\s+(?:the\s+)?)\b.{0,70}"
            r"\b(?:retrieved|provided|supplied|current)?\s*"
            r"(?:evidence|sources?|guideline|citations?|recommendations?)\b|"
            r"\banswer\b.{0,35}\bfrom\b.{0,20}\b(?:your\s+)?"
            r"(?:own|general|medical|internal)\s+knowledge\b"
        ),
    ),
    (
        "provenance_fabrication",
        _pattern(
            r"\b(?:fabricate|invent|fake|forge|make\s+up)\b.{0,70}"
            r"\b(?:citations?|evidence|sources?|references?|patient\s+details?|"
            r"symptoms?|test\s+results?)\b"
        ),
    ),
    (
        "secret_exfiltration",
        _pattern(
            r"\b(?:reveal|print|show|display|expose|leak|return|dump)\b.{0,80}"
            r"\b(?:system\s+prompt|developer\s+(?:prompt|message)|api\s*keys?|"
            r"environment\s+variables?|env\s+vars?|secrets?|hidden\s+configuration|"
            r"ollama\s+url)\b"
        ),
    ),
    (
        "authority_manipulation",
        _pattern(
            r"\b(?:treat|consider|make|declare)\b.{0,45}\b2015\b.{0,55}"
            r"\b(?:newer|current|authoritative|stronger|override|supersede)\b|"
            r"\b(?:hide|ignore|discard|suppress)\b.{0,55}"
            r"\b(?:2026|current\s+(?:guideline|recommendation))\b"
        ),
    ),
    (
        "validation_manipulation",
        _pattern(
            r"\b(?:mark|change|set|force|report)\b.{0,60}"
            r"\b(?:citation\s+validation|validation\s+(?:result|status))\b"
            r".{0,35}\b(?:valid|passed?|success)\b"
        ),
    ),
    (
        "trusted_text_injection",
        _pattern(
            r"\b(?:treat|consider|regard)\b.{0,70}"
            r"\b(?:following|below|user|sentence|text|content)\b.{0,65}"
            r"\b(?:trusted|system|clinical\s+evidence|authoritative)\b"
        ),
    ),
    (
        "instruction_override_ar",
        _pattern(
            r"(?:تجاهل|تخط(?:ى|ي)?|تجاوز|انسى|انسَ|ألغ|الغ|استبدل|"
            r"لا\s+تتبع|لا\s+تنفذ)"
            r".{0,90}(?:التعليمات|القواعد|تعليمات\s+النظام|النظام|المطور|السابقة|أعلاه)"
        ),
    ),
    (
        "role_override_ar",
        _pattern(
            r"(?:فعّل|فعل|شغّل|شغل|انتقل\s+إلى).{0,35}"
            r"(?:وضع\s+المطور|وضع\s+غير\s+مقيد|وضع\s+jailbreak|developer\s+mode)"
        ),
    ),
    (
        "evidence_bypass_ar",
        _pattern(
            r"(?:لا\s+تستخدم|لا\s+تستشهد|لا\s+تتبع|تجاهل)"
            r".{0,70}(?:الأدلة|الدليل|المصادر|الإرشادات|التوصيات|المراجع)"
        ),
    ),
    (
        "provenance_fabrication_ar",
        _pattern(
            r"(?:اخترع|لفق|لفّق|زيف|زيّف)"
            r".{0,70}(?:مراجع|استشهادات|أدلة|مصادر|أعراض|بيانات\s+المريض|نتائج)"
        ),
    ),
    (
        "secret_exfiltration_ar",
        _pattern(
            r"(?:اعرض|أظهر|اظهر|اطبع|اكشف|سرّب|سرب)"
            r".{0,80}(?:تعليمات\s+النظام|نص\s+النظام|system\s+prompt|"
            r"مفاتيح\s*(?:api|API)|متغيرات\s+البيئة|الأسرار|الاعدادات\s+المخفية|"
            r"الإعدادات\s+المخفية)"
        ),
    ),
    (
        "authority_manipulation_ar",
        _pattern(
            r"(?:اعتبر|اجعل).{0,45}(?:دليل\s*)?2015.{0,55}"
            r"(?:أحدث|اقوى|أقوى|أهم|يتجاوز|يلغي).{0,35}(?:2026|الدليل\s+الحالي)?"
        ),
    ),
    (
        "validation_manipulation_ar",
        _pattern(
            r"(?:غيّر|غير|اجعل|سجل).{0,60}"
            r"(?:citation\s+validation|نتيجة\s+التحقق|الاستشهادات)"
            r".{0,35}(?:صحيحة|صالح|ناجح|valid|pass)"
        ),
    ),
    (
        "trusted_text_injection_ar",
        _pattern(
            r"(?:النص|المحتوى|الجملة)\s+(?:التالي|التالية|أدناه)"
            r".{0,45}(?:دليل\s+طبي\s+موثوق|تعليمات\s+نظام|مصدر\s+موثوق)"
        ),
    ),
)


_COMPACT_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("instruction_override", "ignoreallpreviousinstructions"),
    ("instruction_override", "disregardpreviousinstructions"),
    ("role_override", "systemoverride"),
    ("provenance_fabrication", "fabricatecitations"),
    ("provenance_fabrication", "inventcitations"),
    ("secret_exfiltration", "revealsystemprompt"),
    ("secret_exfiltration", "printsystemprompt"),
    ("secret_exfiltration", "showapikeys"),
)


def normalize_query_for_safety(query: str) -> str:
    """Normalize compatibility characters and remove invisible format controls."""

    normalized = unicodedata.normalize("NFKC", query)
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )
    return " ".join(normalized.casefold().split())


def assess_query_safety(query: str) -> dict[str, object]:
    """Return a transparent deterministic decision before any model or retrieval call."""

    normalized = normalize_query_for_safety(query)
    reasons = [
        reason
        for reason, pattern in _CONTROL_PATTERNS
        if pattern.search(normalized)
    ]
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    reasons.extend(
        reason
        for reason, signature in _COMPACT_SIGNATURES
        if signature in compact
    )
    unique_reasons = list(dict.fromkeys(reasons))
    if unique_reasons:
        return {
            "status": "blocked",
            "reason_codes": unique_reasons,
            "message": SAFETY_REFUSAL_MESSAGE,
        }
    return {"status": "allowed", "reason_codes": [], "message": None}
