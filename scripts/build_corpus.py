"""Build the reconciled, section-aware two-source NG12 corpus."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.chunker import build_chunks  # noqa: E402
from src.ingestion.corpus_report import build_merge_report, readable_merge_summary  # noqa: E402
from src.ingestion.full_pdf_parser import parse_full_guideline  # noqa: E402
from src.ingestion.normalize import normalize_current_records  # noqa: E402
from src.ingestion.pdf_parser import parse_pdf_pages  # noqa: E402
from src.ingestion.reconcile import reconcile_records  # noqa: E402
from src.ingestion.report import build_parsing_report  # noqa: E402
from src.ingestion.scope_filter import extract_scoped_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True, help="Current 2026 NG12 PDF")
    parser.add_argument("--full", type=Path, required=True, help="Full 2015 NG12 PDF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "parsed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    current_input = args.current.expanduser().resolve()
    full_input = args.full.expanduser().resolve()
    if not current_input.is_file():
        raise FileNotFoundError(f"Current guideline PDF not found: {current_input}")
    if not full_input.is_file():
        raise FileNotFoundError(f"Full guideline PDF not found: {full_input}")

    current_pdf = _copy_source(current_input, "ng12_current_2026.pdf")
    full_pdf = _copy_source(full_input, "ng12_full_2015.pdf")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    current_pages, current_inspection, current_toc = parse_pdf_pages(current_pdf)
    current_scoped, current_scope_diagnostics = extract_scoped_records(
        current_pdf, current_pages, current_toc
    )
    current_report = build_parsing_report(
        current_pdf,
        current_pages,
        current_scoped,
        current_inspection,
        current_scope_diagnostics,
    )
    current_records = normalize_current_records(current_scoped)

    full_pages, full_records, full_inspection, _ = parse_full_guideline(full_pdf)
    records, reconciliation = reconcile_records(current_records, full_records)
    chunks, chunking = build_chunks(records)
    merge_report = build_merge_report(
        current_pdf=current_pdf,
        full_pdf=full_pdf,
        current_pages=len(current_pages),
        full_pages=len(full_pages),
        current_parsing_report=current_report,
        full_inspection=full_inspection,
        records=records,
        reconciliation=reconciliation,
        chunks=chunks,
        chunking=chunking,
    )

    current_page_rows = []
    for page in current_pages:
        payload = page.to_dict()
        payload.update(
            {
                "source_file": "ng12_current_2026.pdf",
                "source_version": "2026_current",
                "source_type": "current_guideline",
                "authority_priority": "primary",
            }
        )
        current_page_rows.append(payload)

    full_page_rows = []
    for page in full_pages:
        payload = page.to_dict()
        payload.update(
            {
                "source_version": "2015_full",
                "source_type": "full_guideline",
                "authority_priority": "supporting",
            }
        )
        full_page_rows.append(payload)

    _write_jsonl(output_dir / "pages_current.jsonl", current_page_rows)
    _write_jsonl(output_dir / "pages_full.jsonl", full_page_rows)
    _write_jsonl(output_dir / "records_clean.jsonl", [record.to_dict() for record in records])
    _write_jsonl(output_dir / "chunks.jsonl", [chunk.to_dict() for chunk in chunks])
    _write_json(output_dir / "merge_report.json", merge_report)

    print(readable_merge_summary(merge_report))
    return 0


def _copy_source(source: Path, filename: str) -> Path:
    destination = (PROJECT_ROOT / "data" / "raw" / filename).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination:
        shutil.copy2(source, destination)
    return destination


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
