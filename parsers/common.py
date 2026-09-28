"""Shared OCR loading, geometry helpers, dates, and field metadata."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

FieldStatus = Literal["CONFIRMED", "CANDIDATE", "MISSING", "REVIEW_REQUIRED"]

DATE_NUMERIC_RE = re.compile(
    r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b"
)
DATE_COMPACT_RE = re.compile(
    r"\b(\d{1,2})\s*([A-Za-z]{3,9})\s*(\d{4})\b", re.I
)
DATE_COMPACT2_RE = re.compile(r"\b(\d{2})([A-Za-z]{3})(\d{4})\b")
MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


@dataclass
class OcrElement:
    index: int
    text: str
    confidence: float = 0.0
    box: list[float] | None = None

    @property
    def stripped(self) -> str:
        return self.text.strip()

    @property
    def is_empty(self) -> bool:
        return not self.stripped

    @property
    def center(self) -> tuple[float, float] | None:
        if not self.box or len(self.box) < 4:
            return None
        x1, y1, x2, y2 = self.box[0], self.box[1], self.box[2], self.box[3]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class FieldValue:
    value: str | None
    status: FieldStatus = "MISSING"
    source: str = ""
    ocr_confidence: float | None = None
    raw_value: str | None = None
    quality_score: float | None = None

    def to_public(self) -> Any:
        return self.value


@dataclass
class ParseContext:
    elements: list[OcrElement]
    warnings: list[str] = field(default_factory=list)

    def non_empty(self) -> list[OcrElement]:
        return [e for e in self.elements if not e.is_empty]


def load_ocr_json(path: Path | str) -> tuple[dict, list[OcrElement]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    elements: list[OcrElement] = []
    for page in data.get("pages", []):
        for ln in page.get("lines", []):
            elements.append(
                OcrElement(
                    index=int(ln.get("index", len(elements))),
                    text=str(ln.get("text", "")),
                    confidence=float(ln.get("confidence", 0.0) or 0.0),
                    box=ln.get("box"),
                )
            )
    return data, elements


def latest_ocr_json_for_image(samples_path: Path, output_dir: Path) -> Path | None:
    stem = samples_path.stem
    matches = sorted(output_dir.glob(f"{stem}_*_ocr.json"), key=lambda p: p.stat().st_mtime)
    if matches:
        return matches[-1]
    # legacy name image_*.json
    if stem == "image":
        folder = samples_path.parent.name
        all_json = sorted(output_dir.glob("*_ocr.json"), key=lambda p: p.stat().st_mtime)
        for p in reversed(all_json):
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
                src = meta.get("source_image", "")
                if folder in src.replace("\\", "/").lower():
                    return p
            except json.JSONDecodeError:
                continue
    return None


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def mask_passport_number(value: str | None) -> str:
    if not value:
        return "—"
    v = value.replace("<", "").strip()
    if len(v) <= 4:
        return "****"
    return v[:4] + "*" * (len(v) - 6) + v[-2:] if len(v) > 6 else v[:2] + "****"


def mask_name(value: str | None) -> str:
    if not value:
        return "—"
    parts = value.split()
    masked = []
    for p in parts:
        if len(p) <= 1:
            masked.append(p)
        else:
            masked.append(p[0] + "*" * min(3, len(p) - 1))
    return " ".join(masked)


def mask_licence_number(value: str | None) -> str:
    if not value:
        return "—"
    if len(value) <= 4:
        return "****"
    return value[:4] + "*" * (len(value) - 6) + value[-2:]


def parse_date_to_iso(raw: str, kind: str = "generic") -> tuple[str | None, list[str]]:
    """Parse common date strings to YYYY-MM-DD. Returns (iso, warnings)."""
    warnings: list[str] = []
    raw = raw.strip()
    if not raw:
        return None, warnings

    m = DATE_NUMERIC_RE.search(raw)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y = _expand_two_digit_year(y, kind, mo, d, warnings)
        try:
            return date(y, mo, d).isoformat(), warnings
        except ValueError:
            return None, warnings + ["INVALID_DATE"]

    m = DATE_COMPACT_RE.search(raw) or DATE_COMPACT2_RE.search(raw)
    if m:
        d = int(m.group(1))
        mon = MONTHS.get(m.group(2).upper()[:3])
        y = int(m.group(3))
        if mon:
            try:
                return date(y, mon, d).isoformat(), warnings
            except ValueError:
                return None, warnings + ["INVALID_DATE"]

    return None, warnings


def _expand_two_digit_year(y: int, kind: str, month: int, day: int, warnings: list[str]) -> int:
    today = date.today()
    candidates = [1900 + y, 2000 + y]
    valid = []
    for full in candidates:
        try:
            dt = date(full, month, day)
            valid.append(dt)
        except ValueError:
            continue
    if not valid:
        warnings.append("DATE_CENTURY_AMBIGUOUS")
        return candidates[0]
    if kind == "dob":
        past = [d for d in valid if d <= today]
        if past:
            return max(past).year
        warnings.append("DATE_CENTURY_AMBIGUOUS")
        return valid[0].year
    if kind == "expiry":
        future_or_recent = [d for d in valid if d.year >= today.year - 30]
        if future_or_recent:
            return max(future_or_recent).year
        warnings.append("DATE_CENTURY_AMBIGUOUS")
        return valid[-1].year
    chosen = min(valid, key=lambda d: abs((d - today).days))
    if len(valid) > 1 and abs(valid[0].year - valid[1].year) >= 100:
        warnings.append("DATE_CENTURY_AMBIGUOUS")
    return chosen.year


def expand_mrz_yymmdd(yymmdd: str, kind: str) -> tuple[str | None, list[str]]:
    if not re.fullmatch(r"\d{6}", yymmdd):
        return None, ["MRZ_DATE_INVALID"]
    yy, mm, dd = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    warnings: list[str] = []
    year = _expand_two_digit_year(yy, kind, mm, dd, warnings)
    try:
        return date(year, mm, dd).isoformat(), warnings
    except ValueError:
        return None, ["MRZ_DATE_INVALID"]


def label_regex(patterns: list[str]) -> re.Pattern[str]:
    inner = "|".join(patterns)
    return re.compile(rf"(?:{inner})", re.I)


def value_after_label(text: str, label_end: int | None = None) -> str | None:
    if label_end is None:
        if ":" in text:
            return text.split(":", 1)[1].strip() or None
        return None
    return text[label_end:].strip() or None


def find_label_elements(elements: list[OcrElement], pattern: re.Pattern[str]) -> list[OcrElement]:
    return [e for e in elements if pattern.search(e.text)]


def nearest_value_for_label(
    elements: list[OcrElement],
    label_el: OcrElement,
    validator: Any,
    max_index_gap: int = 4,
) -> tuple[OcrElement | None, str]:
    """Return (element, strategy) for value linked to label."""
    m = None
    for pat in (r":\s*(.+)$", r"\s{2,}(.+)$"):
        m = re.search(pat, label_el.text)
        if m and validator(m.group(1).strip()):
            return label_el, "SAME_LINE"

    label_idx = elements.index(label_el)
    for nxt in elements[label_idx + 1 : label_idx + 1 + max_index_gap]:
        if nxt.is_empty:
            continue
        if validator(nxt.stripped):
            return nxt, "NEXT_LINE_INDEX"

    label_c = label_el.center
    if label_c:
        below: list[tuple[float, OcrElement]] = []
        for e in elements:
            if e.is_empty or e.index <= label_el.index:
                continue
            c = e.center
            if not c:
                continue
            if c[1] > label_c[1] and abs(c[0] - label_c[0]) < 200:
                if validator(e.stripped):
                    below.append((c[1] - label_c[1], e))
        if below:
            below.sort(key=lambda t: t[0])
            return below[0][1], "BELOW_GEOMETRY"

    return None, ""


def quality_score(
    ocr_conf: float,
    source: str,
    format_ok: bool,
    cross: bool = False,
) -> float:
    base = max(0.0, min(1.0, ocr_conf))
    bonus = 0.0
    if source in ("MRZ", "MRZ_AND_VISUAL", "LABEL_ADJACENT", "SAME_LINE"):
        bonus += 0.05
    if format_ok:
        bonus += 0.05
    if cross:
        bonus += 0.1
    return min(1.0, base + bonus)
