"""Ingestion utilities for the NICE NG12 source document."""

from .models import PageRecord, ScopedRecord
from .pdf_parser import parse_pdf_pages
from .scope_filter import extract_scoped_records

__all__ = ["PageRecord", "ScopedRecord", "extract_scoped_records", "parse_pdf_pages"]
