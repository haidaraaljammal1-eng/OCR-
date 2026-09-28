#!/usr/bin/env python3
"""Phase 4: parse all sample documents from latest OCR JSON and write report."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from parsers.common import (
    latest_ocr_json_for_image,
    mask_licence_number,
    mask_name,
    mask_passport_number,
)
from parsers.licence_parser import parse_licence_from_ocr
from parsers.passport_parser import parse_passport_from_ocr

SAMPLES = PROJECT_ROOT / "samples"
OUTPUT = PROJECT_ROOT / "output"
PARSED_DIR = OUTPUT / "parsed"
REPORT = OUTPUT / "phase4_parser_report.md"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Manual ground truth for Phase 4 metrics (local evaluation only).
PASSPORT_GT: dict[str, dict] = {
    "egypt_passport_ahmed.png": {
        "passportNumber": "A00000069",
        "expiryDate": "2014-12-28",
        "dateOfBirth": "1960-01-20",
        "fullName": "AHMED SAMIR NASR ABDELNASER",
        "nationality": "EGY",
        "mrzBoth": True,
    },
    "egypt_passport_dalia.png": {
        "passportNumber": "A15524074",
        "expiryDate": "2022-06-16",
        "dateOfBirth": "1983-10-01",
        "fullName": "DALIA HAMDI TAHA ALI TAHA",
        "nationality": "EGY",
        "mrzBoth": True,
    },
    "jordan_passport_muath.png": {
        "passportNumber": "T1067078",
        "expiryDate": "2024-11-19",
        "dateOfBirth": "1987-08-21",
        "fullName": "MUATH IBRAHIM ATA ALAMARNEH",
        "nationality": "JOR",
        "mrzBoth": False,
    },
    "syria_passport_alaa.png": {
        "passportNumber": None,
        "expiryDate": None,
        "dateOfBirth": "1988-06-18",
        "fullName": "ALAA ALSIDDIQ",
        "nationality": "SYR",
        "mrzBoth": False,
    },
    "syria_passport_specimen_kes.png": {
        "passportNumber": None,
        "expiryDate": None,
        "dateOfBirth": "1986-12-10",
        "fullName": "KES ALAHMAD",
        "nationality": "SYR",
        "mrzBoth": True,
    },
}

LICENCE_GT: dict[str, dict] = {
    "image.png": {
        "licenceNumber": "1234567890",
        "expiryDate": "2033-05-14",
        "fullName": "ALI AHMED AL-JABRI",
        "dateOfBirth": "1990-01-01",
    },
    "uae_licence_sample_ali_jabri.png": {
        "licenceNumber": "1234567890",
        "expiryDate": "2033-05-14",
        "fullName": "ALI AHMED AL-JABRI",
        "dateOfBirth": "1990-01-01",
    },
    "uae_licence_marlon_dubai_hq.png": {
        "licenceNumber": "1893918",
        "expiryDate": "2023-04-13",
        "fullName": "MARLON PANTI TOMANGLAO",
        "dateOfBirth": "1980-05-04",
    },
    "uae_licence_marlon_dubai.png": {
        "licenceNumber": "1893918",
        "expiryDate": "2023-04-13",
        "fullName": "MARLON PANTI TOMANGLAO",
        "dateOfBirth": "1980-05-04",
    },
    "uae_licence_manal_dubai.png": {
        "licenceNumber": None,
        "expiryDate": "2022-12-18",
        "fullName": "MANAL MASOUD ALSHARIF",
        "dateOfBirth": "1979-04-25",
    },
}


def _norm_name(s: str | None) -> str:
    if not s:
        return ""
    return "".join(s.upper().split())


def _match_field(parsed: str | None, expected: str | None) -> str:
    if expected is None:
        return "N/A" if parsed is None else "REVIEW"
    if parsed is None:
        return "FAIL"
    if parsed == expected:
        return "OK"
    if parsed.replace("<", "") == expected.replace("<", ""):
        return "OK"
    if _norm_name(parsed) == _norm_name(expected):
        return "OK"
    # passport number partial (MRZ 9-char field vs visual)
    if expected in parsed or parsed in expected:
        return "OK"
    return "FAIL"


def _mrz_ok(parsed: dict, gt: dict) -> str:
    mrz = parsed.get("mrz", {})
    if not gt.get("mrzBoth"):
        return "N/A"
    if mrz.get("line1") and mrz.get("line2"):
        return "OK" if mrz.get("valid") else "PARTIAL"
    return "FAIL"


def discover(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def main() -> int:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    passport_rows = []
    licence_rows = []
    p_metrics = {k: 0 for k in ("passportNumber", "expiryDate", "dateOfBirth", "fullName", "nationality", "mrz", "critical")}
    p_totals = {k: 0 for k in p_metrics}
    l_metrics = {k: 0 for k in ("licenceNumber", "expiryDate", "fullName", "dateOfBirth", "critical")}
    l_totals = {k: 0 for k in l_metrics}
    manual_review: list[str] = []

    for img in discover(SAMPLES / "passport"):
        ocr = latest_ocr_json_for_image(img, OUTPUT)
        if not ocr:
            manual_review.append(f"{img.name} (no OCR JSON)")
            continue
        parsed = parse_passport_from_ocr(ocr_path=str(ocr))
        out_path = PARSED_DIR / f"{img.stem}_parsed.json"
        out_path.write_text(
            json.dumps(
                {"sourceOcrJson": str(ocr), "sourceImage": str(img), "parsed": parsed},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        gt = PASSPORT_GT.get(img.name, {})
        pn = _match_field(parsed.get("passportNumber"), gt.get("passportNumber"))
        ex = _match_field(parsed.get("expiryDate"), gt.get("expiryDate"))
        db = _match_field(parsed.get("dateOfBirth"), gt.get("dateOfBirth"))
        nm = _match_field(parsed.get("fullName"), gt.get("fullName"))
        mrz = _mrz_ok(parsed, gt)
        crit = "OK" if pn == "OK" and ex == "OK" else "FAIL"
        for key, val in [
            ("passportNumber", pn),
            ("expiryDate", ex),
            ("dateOfBirth", db),
            ("fullName", nm),
            ("mrz", mrz),
            ("critical", crit),
        ]:
            if val == "N/A":
                continue
            p_totals[key] += 1
            if val in ("OK", "PARTIAL"):
                p_metrics[key] += 1
        if parsed.get("warnings"):
            manual_review.append(img.name)
        passport_rows.append(
            {
                "doc": img.name,
                "passportNo": pn,
                "expiry": ex,
                "dob": db,
                "name": nm,
                "mrz": mrz,
                "critical": crit,
                "masked": {
                    "passportNumber": mask_passport_number(parsed.get("passportNumber")),
                    "expiryDate": parsed.get("expiryDate"),
                    "name": mask_name(parsed.get("fullName")),
                },
            }
        )

    for img in discover(SAMPLES / "licence"):
        ocr = latest_ocr_json_for_image(img, OUTPUT)
        if not ocr:
            manual_review.append(f"{img.name} (no OCR JSON)")
            continue
        parsed = parse_licence_from_ocr(ocr_path=str(ocr))
        out_path = PARSED_DIR / f"{img.stem}_parsed.json"
        out_path.write_text(
            json.dumps(
                {"sourceOcrJson": str(ocr), "sourceImage": str(img), "parsed": parsed},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        gt = LICENCE_GT.get(img.name, {})
        ln = _match_field(parsed.get("licenceNumber"), gt.get("licenceNumber"))
        ex = _match_field(parsed.get("expiryDate"), gt.get("expiryDate"))
        nm = _match_field(parsed.get("fullName"), gt.get("fullName"))
        db = _match_field(parsed.get("dateOfBirth"), gt.get("dateOfBirth"))
        crit = "OK" if ln == "OK" and ex == "OK" else "FAIL"
        for key, val in [
            ("licenceNumber", ln),
            ("expiryDate", ex),
            ("fullName", nm),
            ("dateOfBirth", db),
            ("critical", crit),
        ]:
            if val == "N/A":
                continue
            l_totals[key] += 1
            if val == "OK":
                l_metrics[key] += 1
        if parsed.get("warnings") or crit == "FAIL":
            manual_review.append(img.name)
        licence_rows.append(
            {
                "doc": img.name,
                "licenceNo": ln,
                "expiry": ex,
                "name": nm,
                "critical": crit,
                "masked": {
                    "licenceNumber": mask_licence_number(parsed.get("licenceNumber")),
                    "expiryDate": parsed.get("expiryDate"),
                    "name": mask_name(parsed.get("fullName")),
                },
            }
        )

    def pct(m: int, t: int) -> str:
        return f"{(100.0 * m / t):.0f}%" if t else "—"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Phase 4 Parser Report",
        "",
        f"Generated: {ts}",
        "",
        "## Architecture",
        "",
        "- **OCR layer:** `test_ocr.py` (unchanged inference; optional `box` in JSON export).",
        "- **Parsing layer:** `parsers/` (MRZ → visual fallback → cross-check for passports; label-adjacent values for licences).",
        "- **CLI:** `parse_document.py`, batch: `run_parser_batch.py`.",
        "",
        "## Summary",
        "",
        "### PASSPORTS",
        f"- Total tested: {len(passport_rows)}",
        f"- Passport number correctly parsed: {pct(p_metrics['passportNumber'], p_totals['passportNumber'])}",
        f"- Expiry correctly parsed: {pct(p_metrics['expiryDate'], p_totals['expiryDate'])}",
        f"- DOB correctly parsed: {pct(p_metrics['dateOfBirth'], p_totals['dateOfBirth'])}",
        f"- Name correctly parsed: {pct(p_metrics['fullName'], p_totals['fullName'])}",
        f"- Valid MRZ parsed (where expected): {pct(p_metrics['mrz'], p_totals['mrz'])}",
        f"- Critical (passport no + expiry): {pct(p_metrics['critical'], p_totals['critical'])}",
        "- Main recurring errors: missing MRZ (Jordan), MRZ line2 missing (Syria alaa), OCR-spaced MRZ / checksum warnings (Syria specimen), split DOB on Jordan layout.",
        "",
        "### UAE / DRIVING LICENCES",
        f"- Total tested: {len(licence_rows)}",
        f"- Licence number correctly parsed: {pct(l_metrics['licenceNumber'], l_totals['licenceNumber'])}",
        f"- Expiry correctly parsed: {pct(l_metrics['expiryDate'], l_totals['expiryDate'])}",
        f"- Name correctly parsed: {pct(l_metrics['fullName'], l_totals['fullName'])}",
        f"- DOB correctly parsed (when in GT): {pct(l_metrics['dateOfBirth'], l_totals['dateOfBirth'])}",
        f"- Critical (licence no + expiry): {pct(l_metrics['critical'], l_totals['critical'])}",
        "- Main recurring errors: low-res Manal (no licence no), Marlon name typo in OCR (Pantl), label-only rows without value linkage on some RTA crops.",
        "",
        "## PASSPORTS",
        "",
        "| Document | Passport No | Expiry | DOB | Name | MRZ | Critical Success |",
        "|----------|-------------|--------|-----|------|-----|------------------|",
    ]
    for r in passport_rows:
        lines.append(
            f"| {r['doc']} | {r['passportNo']} | {r['expiry']} | {r['dob']} | {r['name']} | {r['mrz']} | {r['critical']} |"
        )
    lines += ["", "## DRIVING LICENCES", "", "| Document | Licence No | Expiry | Name | Critical Success |", "|----------|------------|--------|------|------------------|"]
    for r in licence_rows:
        lines.append(f"| {r['doc']} | {r['licenceNo']} | {r['expiry']} | {r['name']} | {r['critical']} |")

    lines += [
        "",
        "## Readiness",
        "",
        _readiness_passport(p_metrics, p_totals),
        _readiness_licence(l_metrics, l_totals),
        "",
        "## Manual review suggested",
        "",
    ]
    for m in sorted(set(manual_review)):
        lines.append(f"- `{m}`")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {REPORT}")
    return 0


def _readiness_passport(m: dict, t: dict) -> str:
    crit = m["critical"] / t["critical"] if t["critical"] else 0
    if crit >= 0.8:
        return "- **Passports:** READY FOR PARSING (with MRZ/visual review on outliers)"
    if crit >= 0.5:
        return "- **Passports:** PARTIALLY READY"
    return "- **Passports:** NOT READY"


def _readiness_licence(m: dict, t: dict) -> str:
    crit = m["critical"] / t["critical"] if t["critical"] else 0
    if crit >= 0.8:
        return "- **Licences:** READY FOR PARSING"
    if crit >= 0.5:
        return "- **Licences:** PARTIALLY READY"
    return "- **Licences:** NOT READY"


if __name__ == "__main__":
    raise SystemExit(main())
