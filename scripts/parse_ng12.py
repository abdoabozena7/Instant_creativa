"""Parse the complete NG12 PDF, then create the narrow-scope evidence records."""

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

from src.ingestion.pdf_parser import parse_pdf_pages  # noqa: E402
from src.ingestion.report import build_parsing_report, readable_summary  # noqa: E402
from src.ingestion.scope_filter import extract_scoped_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "ng12.pdf",
        help="Source NG12 PDF. External inputs are copied to data/raw/ng12.pdf.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "parsed",
        help="Directory for pages.jsonl, scoped_records.jsonl, and parsing_report.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"NG12 PDF not found: {source}")

    raw_destination = (PROJECT_ROOT / "data" / "raw" / "ng12.pdf").resolve()
    raw_destination.parent.mkdir(parents=True, exist_ok=True)
    if source != raw_destination:
        shutil.copy2(source, raw_destination)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pages, inspection, toc = parse_pdf_pages(raw_destination)
    records, scope_diagnostics = extract_scoped_records(raw_destination, pages, toc)
    report = build_parsing_report(
        raw_destination, pages, records, inspection, scope_diagnostics
    )

    _write_jsonl(output_dir / "pages.jsonl", [page.to_dict() for page in pages])
    _write_jsonl(
        output_dir / "scoped_records.jsonl",
        [record.to_dict() for record in records],
    )
    _write_json(output_dir / "parsing_report.json", report)
    print(readable_summary(report))
    return 0


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
