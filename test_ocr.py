#!/usr/bin/env python3
"""Local PaddleOCR smoke/test script — raw text + confidence only (no parsing)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Skip slow multi-hoster connectivity probe; models download on first run if missing.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def _die(message: str, code: int = 1) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def _validate_image_path(image_path: Path) -> None:
    if not image_path.exists():
        _die(f"Image file not found: {image_path}")
    if not image_path.is_file():
        _die(f"Path is not a file: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        _die(
            f"Unsupported image format '{image_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            img.verify()
    except Exception as exc:  # noqa: BLE001 — show clear user-facing message
        _die(f"Cannot open or verify image: {exc}")


def _load_ocr_engine():
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        _die(
            "Failed to import paddleocr. Activate .venv and install dependencies "
            "(see README.md). "
            f"Details: {exc}"
        )

    try:
        return PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            engine="paddle",
        )
    except Exception as exc:  # noqa: BLE001
        _die(f"Failed to load PaddleOCR models (CPU): {exc}")


def _extract_lines(result_page: dict) -> list[dict]:
    texts = result_page.get("rec_texts") or []
    scores = result_page.get("rec_scores")
    boxes = result_page.get("rec_boxes")
    lines: list[dict] = []
    for i, text in enumerate(texts):
        entry: dict = {"index": i, "text": text}
        if scores is not None and i < len(scores):
            try:
                entry["confidence"] = float(scores[i])
            except (TypeError, ValueError):
                entry["confidence"] = scores[i]
        if boxes is not None and i < len(boxes):
            try:
                box = boxes[i]
                entry["box"] = [float(v) for v in box]
            except (TypeError, ValueError):
                pass
        lines.append(entry)
    return lines


_OCR_ENGINE = None


def get_ocr_engine():
    """Reuse a single PaddleOCR instance (same settings as CLI)."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        _OCR_ENGINE = _load_ocr_engine()
    return _OCR_ENGINE


def ocr_image_pages(image_path: Path) -> list[dict]:
    """Run OCR and return page line structures (no console output)."""
    _validate_image_path(image_path)
    ocr = get_ocr_engine()
    raw_results = ocr.predict(str(image_path))
    if not raw_results:
        return []
    pages: list[dict] = []
    for page_idx, page in enumerate(raw_results):
        page_data = dict(page)
        lines = _extract_lines(page_data)
        pages.append({"page_index": page_idx, "lines": lines})
    return pages


def write_ocr_json(
    image_path: Path,
    pages: list[dict],
    out_path: Path | None = None,
    source_image: Path | str | None = None,
) -> Path:
    if out_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = OUTPUT_DIR / f"{image_path.stem}_{stamp}_ocr.json"
    payload = {
        "source_image": str(source_image or image_path.resolve()),
        "engine": "paddle",
        "pages": pages,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return out_path


def _save_json(image_path: Path, pages: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUTPUT_DIR / f"{image_path.stem}_{stamp}_ocr.json"
    payload = {
        "source_image": str(image_path.resolve()),
        "engine": "paddle",
        "pages": pages,
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return out_path


def run_ocr(image_path: Path) -> int:
    _validate_image_path(image_path)

    print(f"Image: {image_path.resolve()}")
    print("Loading PaddleOCR (CPU, Paddle engine)...")
    ocr = _load_ocr_engine()

    print("Running OCR...")
    try:
        raw_results = ocr.predict(str(image_path))
    except Exception as exc:  # noqa: BLE001
        _die(f"OCR inference failed: {exc}")

    if not raw_results:
        print("No OCR results returned.")
        return 0

    all_pages: list[dict] = []
    for page_idx, page in enumerate(raw_results):
        page_data = dict(page)
        lines = _extract_lines(page_data)
        all_pages.append({"page_index": page_idx, "lines": lines})

        print(f"\n--- Page {page_idx} ---")
        if not lines:
            print("(no text detected)")
            continue
        for line in lines:
            conf = line.get("confidence")
            if conf is not None:
                print(f"[{line['index']}] {line['text']}  (confidence: {conf:.4f})")
            else:
                print(f"[{line['index']}] {line['text']}")

    json_path = _save_json(image_path, all_pages)
    print(f"\nJSON saved: {json_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run local PaddleOCR on an image and print detected text with confidence."
    )
    parser.add_argument("image", type=Path, help="Path to the image file")
    args = parser.parse_args()
    return run_ocr(args.image)


if __name__ == "__main__":
    raise SystemExit(main())
