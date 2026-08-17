"""A small dependency-free BM25 implementation for the 440-chunk corpus."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", re.IGNORECASE)
# Keep this deliberately small. A broad English stopword list improved the
# reported terse query but regressed an existing safety-netting benchmark.
# These three high-frequency function words contributed no clinical meaning
# in the observed failure; all other terms retain the frozen BM25 behavior.
QUERY_STOPWORDS = {"about", "the", "what"}


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(document)) for document in documents]
        self.document_lengths = np.asarray(
            [sum(frequencies.values()) for frequencies in self.term_frequencies], dtype=np.float32
        )
        self.average_length = float(self.document_lengths.mean()) if documents else 0.0
        document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequency.update(frequencies.keys())
        count = len(documents)
        self.inverse_document_frequency = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.term_frequencies), dtype=np.float32)
        # Query-only filtering prevents a minimal measured question scaffold
        # from outranking the actual clinical feature. Document tokens remain
        # untouched, so medical terms and recommendation IDs retain behavior.
        query_terms = Counter(
            token for token in tokenize(query) if token not in QUERY_STOPWORDS
        )
        if not query_terms or not self.average_length:
            return scores
        for index, frequencies in enumerate(self.term_frequencies):
            length_normalizer = self.k1 * (
                1 - self.b + self.b * self.document_lengths[index] / self.average_length
            )
            score = 0.0
            for term, query_frequency in query_terms.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                score += (
                    self.inverse_document_frequency.get(term, 0.0)
                    * (frequency * (self.k1 + 1))
                    / (frequency + length_normalizer)
                    * min(query_frequency, 2)
                )
            scores[index] = score
        return scores
