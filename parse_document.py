#!/usr/bin/env python3
"""Parse structured fields from existing PaddleOCR JSON (no OCR re-run)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from parsers.licence_parser import parse_licence_from_ocr
from parsers.passport_parser import parse_passport_from_ocr


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse OCR JSON into structured document fields.")
    parser.add_argument("ocr_json", type=Path, help="Path to *_ocr.json from test_ocr.py")
    parser.add_argument(
        "--type",
        choices=("passport", "licence", "auto"),
        default="auto",
        help="Document type (auto uses source_image path)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output parsed JSON path (default: output/parsed/<stem>_parsed.json)",
    )
    args = parser.parse_args()

    if not args.ocr_json.exists():
        print(f"ERROR: file not found: {args.ocr_json}", file=sys.stderr)
        return 1

    meta = json.loads(args.ocr_json.read_text(encoding="utf-8"))
    src = meta.get("source_image", "").lower()
    doc_type = args.type
    if doc_type == "auto":
        if "passport" in src or "passport" in str(args.ocr_json).lower():
            doc_type = "passport"
        elif "licence" in src or "license" in src:
            doc_type = "licence"
        else:
            doc_type = "passport"

    if doc_type == "passport":
        result = parse_passport_from_ocr(ocr_path=str(args.ocr_json))
    else:
        result = parse_licence_from_ocr(ocr_path=str(args.ocr_json))

    out = args.output
    if not out:
        parsed_dir = PROJECT_ROOT / "output" / "parsed"
        parsed_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(src).stem if src else args.ocr_json.stem.replace("_ocr", "")
        out = parsed_dir / f"{stem}_parsed.json"

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sourceOcrJson": str(args.ocr_json.resolve()),
        "sourceImage": meta.get("source_image"),
        "parsed": result,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Parsed JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
