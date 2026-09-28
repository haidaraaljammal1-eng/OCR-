#!/usr/bin/env python3
"""Phase 5: preprocessing + selective OCR retry for failed/review documents."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from parsers.common import latest_ocr_json_for_image, load_ocr_json
from parsers.licence_parser import parse_licence_from_ocr
from parsers.passport_parser import parse_passport_from_ocr
from preprocessing.image_preprocessor import generate_variants
from test_ocr import ocr_image_pages, write_ocr_json

OUTPUT = PROJECT_ROOT / "output"
PREPROCESSED = OUTPUT / "preprocessed"
PHASE5_JSON = OUTPUT / "phase5_ocr"
REPORT = OUTPUT / "phase5_preprocessing_report.md"

FOCUS_PASSPORTS = [
    "jordan_passport_muath.png",
    "syria_passport_alaa.png",
    "syria_passport_specimen_kes.png",
]
FOCUS_LICENCES = ["uae_licence_manal_dubai.png"]

PASSPORT_GT = {
    "jordan_passport_muath.png": {"passportNumber": "T1067078", "expiryDate": "2024-11-19"},
    "syria_passport_alaa.png": {"passportNumber": None, "expiryDate": None},
    "syria_passport_specimen_kes.png": {"passportNumber": None, "expiryDate": None},
}
LICENCE_GT = {
    "uae_licence_manal_dubai.png": {"licenceNumber": None, "expiryDate": "2022-12-18"},
}


@dataclass
class EvalResult:
    label: str
    parsed: dict
    ocr_json: Path
    score: float
    critical_ok: bool
    mrz_valid: bool
    avg_conf: float


def _avg_confidence(ocr_path: Path) -> float:
    _, elements = load_ocr_json(ocr_path)
    vals = [e.confidence for e in elements if e.stripped]
    return sum(vals) / len(vals) if vals else 0.0


def _critical_passport(parsed: dict, filename: str) -> bool:
    gt = PASSPORT_GT.get(filename, {})
    pn = parsed.get("passportNumber")
    ex = parsed.get("expiryDate")
    if gt.get("passportNumber") is None and gt.get("expiryDate") is None:
        return bool(pn or ex)
    ok_pn = True if gt.get("passportNumber") is None else pn == gt["passportNumber"]
    ok_ex = True if gt.get("expiryDate") is None else ex == gt["expiryDate"]
    return ok_pn and ok_ex


def _critical_licence(parsed: dict, filename: str) -> bool:
    gt = LICENCE_GT.get(filename, {})
    ln = parsed.get("licenceNumber")
    ex = parsed.get("expiryDate")
    if gt.get("licenceNumber") is None:
        ok_ln = True
    else:
        ok_ln = ln == gt["licenceNumber"]
    ok_ex = True if gt.get("expiryDate") is None else ex == gt["expiryDate"]
    return ok_ln and ok_ex


def _score_passport(parsed: dict) -> float:
    score = 0.0
    if parsed.get("passportNumber"):
        score += 35
    if parsed.get("expiryDate"):
        score += 35
    mrz = parsed.get("mrz") or {}
    if mrz.get("line1"):
        score += 10
    if mrz.get("line2"):
        score += 15
    if mrz.get("valid"):
        score += 25
    score -= 2 * len(parsed.get("warnings") or [])
    fq = parsed.get("fieldQuality") or {}
    for k in ("passportNumber", "expiryDate"):
        if k in fq and fq[k].get("score"):
            score += float(fq[k]["score"]) * 5
    return score


def _score_licence(parsed: dict) -> float:
    score = 0.0
    if parsed.get("licenceNumber"):
        score += 45
    if parsed.get("expiryDate"):
        score += 45
    if parsed.get("dateOfBirth"):
        score += 5
    score -= 2 * len(parsed.get("warnings") or [])
    return score


def _merge_mrz_lines(base_pages: list[dict], mrz_pages: list[dict]) -> list[dict]:
    if not base_pages:
        base_pages = [{"page_index": 0, "lines": []}]
    lines = list(base_pages[0].get("lines", []))
    start_idx = max((ln.get("index", 0) for ln in lines), default=-1) + 1
    for page in mrz_pages:
        for ln in page.get("lines", []):
            text = str(ln.get("text", "")).strip()
            if not text:
                continue
            if any(text in str(x.get("text", "")) for x in lines):
                continue
            lines.append(
                {
                    "index": start_idx,
                    "text": text,
                    "confidence": float(ln.get("confidence", 0.0)),
                    "box": ln.get("box"),
                    "source": "mrz_crop_ocr",
                }
            )
            start_idx += 1
    merged = [{"page_index": 0, "lines": lines}]
    return merged


def _evaluate_passport(ocr_path: Path, label: str, filename: str) -> EvalResult:
    parsed = parse_passport_from_ocr(ocr_path=str(ocr_path))
    mrz = parsed.get("mrz") or {}
    return EvalResult(
        label=label,
        parsed=parsed,
        ocr_json=ocr_path,
        score=_score_passport(parsed),
        critical_ok=_critical_passport(parsed, filename),
        mrz_valid=bool(mrz.get("valid")),
        avg_conf=_avg_confidence(ocr_path),
    )


def _evaluate_licence(ocr_path: Path, label: str, filename: str) -> EvalResult:
    parsed = parse_licence_from_ocr(ocr_path=str(ocr_path))
    return EvalResult(
        label=label,
        parsed=parsed,
        ocr_json=ocr_path,
        score=_score_licence(parsed),
        critical_ok=_critical_licence(parsed, filename),
        mrz_valid=False,
        avg_conf=_avg_confidence(ocr_path),
    )


def _run_ocr_variant(image: Path, source_image: Path, tag: str) -> Path:
    pages = ocr_image_pages(image)
    out = PHASE5_JSON / f"{source_image.stem}_{tag}_ocr.json"
    return write_ocr_json(image, pages, out_path=out, source_image=source_image)


def _pick_best(candidates: list[EvalResult]) -> EvalResult:
    return max(
        candidates,
        key=lambda c: (c.critical_ok, c.score, c.mrz_valid, c.avg_conf),
    )


def _status_summary(ev: EvalResult, kind: str) -> str:
    if kind == "passport":
        pn = ev.parsed.get("passportNumber")
        ex = ev.parsed.get("expiryDate")
        mrz = ev.parsed.get("mrz") or {}
        return (
            f"pn={'yes' if pn else 'no'}, exp={'yes' if ex else 'no'}, "
            f"mrz_valid={mrz.get('valid')}, L1={'yes' if mrz.get('line1') else 'no'}, "
            f"L2={'yes' if mrz.get('line2') else 'no'}"
        )
    return (
        f"lic={'yes' if ev.parsed.get('licenceNumber') else 'no'}, "
        f"exp={'yes' if ev.parsed.get('expiryDate') else 'no'}"
    )


def process_passport(image_path: Path) -> tuple[EvalResult, EvalResult, str]:
    filename = image_path.name
    before_json = latest_ocr_json_for_image(image_path, OUTPUT)
    if not before_json:
        before_json = _run_ocr_variant(image_path, image_path, "original")
    before = _evaluate_passport(before_json, "before", filename)

    if before.critical_ok and before.mrz_valid:
        return before, before, "Skipped retry: critical fields OK and MRZ valid."

    candidates = [before]
    variants = generate_variants(image_path, PREPROCESSED, "passport")
    mrz_variant = next((v for v in variants if v.name == "mrz_crop"), None)
    full_variants = [v for v in variants if v.name != "mrz_crop"]

    best_full = before
    for v in full_variants:
        ocr_path = _run_ocr_variant(v.path, image_path, v.name)
        ev = _evaluate_passport(ocr_path, v.name, filename)
        candidates.append(ev)
        if ev.score > best_full.score:
            best_full = ev

    merged_pages = ocr_image_pages(
        mrz_variant.path if mrz_variant else image_path
    ) if mrz_variant else []
    if mrz_variant:
        base_json = best_full.ocr_json
        base_pages = json.loads(base_json.read_text(encoding="utf-8"))["pages"]
        mrz_pages = ocr_image_pages(mrz_variant.path)
        merged = _merge_mrz_lines(base_pages, mrz_pages)
        merged_path = PHASE5_JSON / f"{image_path.stem}_merged_mrz_ocr.json"
        write_ocr_json(image_path, merged, out_path=merged_path, source_image=image_path)
        candidates.append(_evaluate_passport(merged_path, "merged_mrz", filename))

    after = _pick_best(candidates)
    note = f"Best variant: {after.label} (score={after.score:.1f})"
    if after.ocr_json != before.ocr_json:
        out_parsed = OUTPUT / "parsed" / f"{image_path.stem}_phase5_parsed.json"
        out_parsed.write_text(
            json.dumps({"parsed": after.parsed, "ocrJson": str(after.ocr_json)}, indent=2),
            encoding="utf-8",
        )
    return before, after, note


def process_licence(image_path: Path) -> tuple[EvalResult, EvalResult, str]:
    filename = image_path.name
    before_json = latest_ocr_json_for_image(image_path, OUTPUT)
    if not before_json:
        before_json = _run_ocr_variant(image_path, image_path, "original")
    before = _evaluate_licence(before_json, "before", filename)

    if before.critical_ok:
        return before, before, "Skipped retry: critical fields OK."

    candidates = [before]
    for v in generate_variants(image_path, PREPROCESSED, "licence"):
        ocr_path = _run_ocr_variant(v.path, image_path, v.name)
        candidates.append(_evaluate_licence(ocr_path, v.name, filename))

    after = _pick_best(candidates)
    note = f"Best variant: {after.label} (score={after.score:.1f})"
    return before, after, note


def main() -> int:
    PHASE5_JSON.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    p_crit_before = p_crit_after = 0
    l_crit_before = l_crit_after = 0
    p_total = l_total = 0
    mrz_improved = []

    for name in FOCUS_PASSPORTS:
        path = PROJECT_ROOT / "samples" / "passport" / name
        before, after, note = process_passport(path)
        p_total += 1
        p_crit_before += int(before.critical_ok)
        p_crit_after += int(after.critical_ok)
        if (not before.mrz_valid) and after.mrz_valid:
            mrz_improved.append(name)
        rows.append(
            f"| {name} | {before.critical_ok} ({_status_summary(before, 'passport')}) | "
            f"{after.critical_ok} ({_status_summary(after, 'passport')}) | "
            f"{'OK' if after.critical_ok else 'FAIL'} | {note} |"
        )

    for name in FOCUS_LICENCES:
        path = PROJECT_ROOT / "samples" / "licence" / name
        before, after, note = process_licence(path)
        l_total += 1
        l_crit_before += int(before.critical_ok)
        l_crit_after += int(after.critical_ok)
        rows.append(
            f"| {name} | {before.critical_ok} ({_status_summary(before, 'licence')}) | "
            f"{after.critical_ok} ({_status_summary(after, 'licence')}) | "
            f"{'OK' if after.critical_ok else 'FAIL'} | {note} |"
        )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = [
        "# Phase 5 — Preprocessing + Selective OCR Retry",
        "",
        f"Generated: {ts}",
        "",
        "## Comparison",
        "",
        "| Document | Before | After | Critical Success | Notes |",
        "|----------|--------|-------|------------------|-------|",
        *rows,
        "",
        "## Metrics",
        "",
        f"- Passport critical success: **{p_crit_before}/{p_total}** before → **{p_crit_after}/{p_total}** after",
        f"- Licence critical success: **{l_crit_before}/{l_total}** before → **{l_crit_after}/{l_total}** after",
        f"- MRZ validity improved on: {', '.join(mrz_improved) if mrz_improved else 'none'}",
        "- Manal licence: see row above for licence number / expiry extraction after retry.",
        "",
        "Preprocessed images: `output/preprocessed/`",
        "Phase 5 OCR JSON: `output/phase5_ocr/`",
        "",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(f"Report written: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
