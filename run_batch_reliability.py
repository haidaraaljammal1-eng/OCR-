#!/usr/bin/env python3
"""Batch-run test_ocr.py and build output/batch_reliability_report.md (Phase 3)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PASSPORT_DIR = PROJECT_ROOT / "samples" / "passport"
LICENCE_DIR = PROJECT_ROOT / "samples" / "licence"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORT_PATH = OUTPUT_DIR / "batch_reliability_report.md"
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
TEST_OCR = PROJECT_ROOT / "test_ocr.py"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

MRZ_LINE1_RE = re.compile(r"^P[<A-Z0-9].{20,}$")
MRZ_LINE2_RE = re.compile(r"^[A-Z0-9<]{25,}$")
DATE_RE = re.compile(
    r"\b(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{2}[/.-]\d{2}[/.-]\d{4})\b"
)
PASSPORT_NO_RE = re.compile(r"\b[A-Z]?\d{6,9}\b")
LICENCE_NO_RE = re.compile(r"(?:LICENSE|LICENCE)\s*NO\.?\s*:?\s*(\d{4,12})", re.I)


@dataclass
class FieldEval:
    status: str
    confidence: str
    notes: str
    evidence: str = ""


@dataclass
class DocResult:
    category: str
    filename: str
    json_path: Path
    lines: list[dict]
    fields: dict[str, FieldEval] = field(default_factory=dict)
    avg_confidence_nonempty: float = 0.0
    mrz_notes: str = ""


def discover_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def run_test_ocr(image: Path) -> Path:
    env = os.environ.copy()
    env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [str(PYTHON), str(TEST_OCR), str(image)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"test_ocr failed for {image.name}:\n{proc.stdout}\n{proc.stderr}"
        )
    for line in proc.stdout.splitlines():
        if "JSON saved:" in line:
            path_str = line.split("JSON saved:", 1)[1].strip()
            return Path(path_str)
    raise RuntimeError(f"No JSON path in output for {image}")


def load_lines(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    lines: list[dict] = []
    for page in data.get("pages", []):
        lines.extend(page.get("lines", []))
    return lines


def nonempty_confidence(lines: list[dict]) -> float:
    vals = [ln["confidence"] for ln in lines if str(ln.get("text", "")).strip()]
    return sum(vals) / len(vals) if vals else 0.0


def all_text(lines: list[dict]) -> str:
    return "\n".join(str(ln.get("text", "")) for ln in lines)


def find_mrz_lines(lines: list[dict]) -> tuple[str | None, str | None, float, float]:
    l1 = l2 = None
    c1 = c2 = 0.0
    for ln in lines:
        t = str(ln.get("text", "")).strip()
        if not t:
            continue
        if MRZ_LINE1_RE.match(t.replace(" ", "")):
            l1, c1 = t, float(ln.get("confidence", 0))
        elif MRZ_LINE2_RE.match(t.replace(" ", "")) and "<" in t:
            l2, c2 = t, float(ln.get("confidence", 0))
    return l1, l2, c1, c2


def status_present(conf: float, found: bool, minor_issue: bool = False) -> str:
    if not found:
        return "NOT READ"
    if minor_issue:
        return "READ WITH MINOR ERROR"
    # No ground truth — value detected only
    return "OCR VALUE PRESENT"


def eval_passport(doc: DocResult) -> None:
    lines = doc.lines
    blob = all_text(lines)
    blob_u = blob.upper()

    l1, l2, c1, c2 = find_mrz_lines(lines)
    mrz_split_bad = False
    mrz_notes: list[str] = []
    if l1:
        if " " in l1.strip() and l1.count(" ") > 3:
            mrz_split_bad = True
            mrz_notes.append("MRZ line 1 contains excessive spaces (possible character splitting).")
    else:
        mrz_notes.append("MRZ line 1 not found as a single cohesive line.")
    if l2:
        if " " in l2.strip() and l2.count(" ") > 5:
            mrz_split_bad = True
            mrz_notes.append("MRZ line 2 contains excessive spaces (possible character splitting).")
    else:
        mrz_notes.append("MRZ line 2 not found as a single cohesive line.")
    if l1 and l2:
        mrz_notes.append("Both MRZ lines detected as continuous strings.")
    doc.mrz_notes = " ".join(mrz_notes)

    # Full name
    name_line = next(
        (ln for ln in lines if re.search(r"\bNAME\b|الاسم|الإسم", str(ln.get("text", "")), re.I)),
        None,
    )
    name_found = bool(
        name_line
        or re.search(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,}){2,}\b", blob_u)
        or (l1 and "<<" in l1)
    )
    name_conf = float(name_line["confidence"]) if name_line else (c1 if l1 else 0.0)
    doc.fields["Full Name"] = FieldEval(
        status_present(name_conf, name_found),
        f"{name_conf:.4f}" if name_found else "—",
        "Manual verification required; no ground truth in batch.",
        (name_line or {}).get("text", l1 or "")[:120],
    )

    # Passport number
    pno_line = None
    for ln in lines:
        t = str(ln.get("text", ""))
        if re.search(r"passport|جواز|رقم", t, re.I) and PASSPORT_NO_RE.search(t):
            pno_line = ln
            break
    pno = None
    if pno_line:
        m = PASSPORT_NO_RE.search(str(pno_line.get("text", "")))
        pno = m.group(0) if m else None
    if not pno and l2:
        m = re.match(r"^([A-Z0-9<]{9,10})", l2.replace(" ", ""))
        if m:
            pno = m.group(1).replace("<", "").strip() or None
    if not pno:
        for ln in lines:
            t = str(ln.get("text", "")).strip()
            if re.fullmatch(r"[A-Z]\d{7,8}", t) or re.fullmatch(r"\d{8}", t):
                pno = t
                pno_line = ln
                break
    pno_conf = float(pno_line["confidence"]) if pno_line else (c2 if pno else 0.0)
    doc.fields["Passport Number"] = FieldEval(
        status_present(pno_conf, bool(pno)),
        f"{pno_conf:.4f}" if pno else "—",
        "Manual verification required.",
        pno or "",
    )

    # Nationality
    nat_line = next(
        (
            ln
            for ln in lines
            if re.search(r"nationality|جنسية|EGYPTIAN|JORDANIAN|SYRIAN", str(ln.get("text", "")), re.I)
            or re.search(r"\b(EGY|JOR|SYR)\b", str(ln.get("text", "")))
        ),
        None,
    )
    nat_found = bool(nat_line) or bool(l2 and re.search(r"[A-Z]{3}", l2))
    nat_conf = float(nat_line["confidence"]) if nat_line else 0.0
    doc.fields["Nationality"] = FieldEval(
        status_present(nat_conf, nat_found),
        f"{nat_conf:.4f}" if nat_found else "—",
        "May appear as country code or label; verify manually.",
        (nat_line or {}).get("text", "")[:120],
    )

    # DOB
    dob_line = next(
        (ln for ln in lines if re.search(r"birth|ميلاد", str(ln.get("text", "")), re.I)),
        None,
    )
    dob_val = None
    for ln in lines:
        t = str(ln.get("text", ""))
        if DATE_RE.search(t) and (dob_line is None or ln == dob_line or "birth" in t.lower()):
            dob_val = DATE_RE.search(t)
            if dob_val and "expiry" not in t.lower() and "issue" not in t.lower():
                dob_line = ln
                break
    if not dob_val:
        for ln in lines:
            t = str(ln.get("text", ""))
            m = DATE_RE.search(t)
            if m and "expiry" not in t.lower() and "issue" not in t.lower():
                dob_val = m
                dob_line = ln
    dob_found = dob_val is not None or (l2 is not None)
    dob_conf = float(dob_line["confidence"]) if dob_line else 0.0
    doc.fields["Date of Birth"] = FieldEval(
        status_present(dob_conf, dob_found),
        f"{dob_conf:.4f}" if dob_found else "—",
        "Date-like token in OCR; MRZ may also encode DOB.",
        (dob_line or {}).get("text", "")[:80],
    )

    # Expiry
    exp_line = next(
        (ln for ln in lines if re.search(r"expiry|انتهاء|انتهاء", str(ln.get("text", "")), re.I)),
        None,
    )
    exp_val = None
    for ln in lines:
        t = str(ln.get("text", ""))
        if "expiry" in t.lower() or "انتهاء" in t:
            m = DATE_RE.search(t)
            if m:
                exp_line = ln
                exp_val = m
                break
    if not exp_val:
        # last date lines often expiry in passport layout — still OCR VALUE PRESENT only
        dates = [(ln, DATE_RE.search(str(ln.get("text", "")))) for ln in lines]
        dates = [(ln, m) for ln, m in dates if m]
        if dates:
            exp_line, exp_val = dates[-1]
    exp_found = exp_val is not None or (l2 is not None)
    exp_conf = float(exp_line["confidence"]) if exp_line else 0.0
    doc.fields["Date of Expiry"] = FieldEval(
        status_present(exp_conf, exp_found),
        f"{exp_conf:.4f}" if exp_found else "—",
        "Verify against visual zone vs MRZ manually.",
        (exp_line or {}).get("text", "")[:80],
    )

    minor_mrz = mrz_split_bad
    doc.fields["MRZ line 1"] = FieldEval(
        status_present(c1, l1 is not None, minor_mrz),
        f"{c1:.4f}" if l1 else "—",
        doc.mrz_notes,
        (l1 or "")[:100],
    )
    doc.fields["MRZ line 2"] = FieldEval(
        status_present(c2, l2 is not None, minor_mrz),
        f"{c2:.4f}" if l2 else "—",
        doc.mrz_notes,
        (l2 or "")[:100],
    )


def eval_licence(doc: DocResult) -> None:
    lines = doc.lines
    blob = all_text(lines)

    def line_with(pattern: str) -> dict | None:
        return next(
            (ln for ln in lines if re.search(pattern, str(ln.get("text", "")), re.I)),
            None,
        )

    name_ln = line_with(r"\bNAME\b|الاسم")
    name_found = name_ln is not None
    doc.fields["Name"] = FieldEval(
        status_present(float(name_ln["confidence"]) if name_ln else 0, name_found),
        f"{float(name_ln['confidence']):.4f}" if name_ln else "—",
        "English name line; manual verification required.",
        str(name_ln.get("text", ""))[:120] if name_ln else "",
    )

    lic_ln = line_with(r"LICENSE\s*NO|LICENCE\s*NO|رقم الرخصة")
    lic_no = None
    if lic_ln:
        m = LICENCE_NO_RE.search(str(lic_ln.get("text", "")))
        if m:
            lic_no = m.group(1)
        else:
            m2 = re.search(r"\d{4,12}", str(lic_ln.get("text", "")))
            lic_no = m2.group(0) if m2 else None
    if not lic_no:
        for ln in lines:
            t = str(ln.get("text", ""))
            if re.search(r"^\d{6,10}$", t.strip()):
                lic_no = t.strip()
                lic_ln = ln
                break
    doc.fields["Licence Number"] = FieldEval(
        status_present(float(lic_ln["confidence"]) if lic_ln else 0, bool(lic_no)),
        f"{float(lic_ln['confidence']):.4f}" if lic_ln and lic_no else "—",
        "Manual verification required.",
        lic_no or "",
    )

    exp_ln = line_with(r"EXPIRY|انتهاء")
    exp_date = None
    if exp_ln:
        m = DATE_RE.search(str(exp_ln.get("text", "")))
        exp_date = m.group(0) if m else None
    doc.fields["Date of Expiry"] = FieldEval(
        status_present(float(exp_ln["confidence"]) if exp_ln else 0, bool(exp_date)),
        f"{float(exp_ln['confidence']):.4f}" if exp_ln and exp_date else "—",
        "English expiry field.",
        str(exp_ln.get("text", ""))[:80] if exp_ln else "",
    )

    dob_ln = line_with(r"BIRTH|ميلاد")
    dob_date = DATE_RE.search(str(dob_ln.get("text", ""))) if dob_ln else None
    if dob_ln:
        st = status_present(float(dob_ln["confidence"]), bool(dob_date))
    else:
        st = "NOT READ"
    doc.fields["Date of Birth"] = FieldEval(
        st,
        f"{float(dob_ln['confidence']):.4f}" if dob_ln and dob_date else "—",
        "Optional; present if labelled in OCR.",
        str(dob_ln.get("text", ""))[:80] if dob_ln else "",
    )

    nat_ln = line_with(r"NATIONALITY|جنسية")
    nat_found = nat_ln is not None
    doc.fields["Nationality"] = FieldEval(
        status_present(float(nat_ln["confidence"]) if nat_ln else 0, nat_found),
        f"{float(nat_ln['confidence']):.4f}" if nat_ln else "—",
        "Optional; English line sufficient per test scope.",
        str(nat_ln.get("text", ""))[:80] if nat_ln else "",
    )


def classify_readiness(docs: list[DocResult], kind: str) -> str:
    if not docs:
        return "NOT READY"
    if kind == "passport":
        critical = ["Passport Number", "Date of Expiry", "MRZ line 1", "MRZ line 2"]
    else:
        critical = ["Name", "Licence Number", "Date of Expiry"]

    ok = 0
    for doc in docs:
        if all(
            doc.fields[f].status in ("OCR VALUE PRESENT", "READ CORRECTLY", "READ WITH MINOR ERROR")
            for f in critical
            if f in doc.fields
        ):
            ok += 1
    ratio = ok / len(docs)
    if ratio >= 0.8:
        return "READY FOR PARSING"
    if ratio >= 0.5:
        return "PARTIALLY READY"
    return "NOT READY"


def pct(docs: list[DocResult], field: str) -> float:
    if not docs:
        return 0.0
    good = sum(
        1
        for d in docs
        if d.fields.get(field)
        and d.fields[field].status
        not in ("NOT READ", "UNCLEAR")
    )
    return 100.0 * good / len(docs)


def build_report(passports: list[DocResult], licences: list[DocResult]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    p_avg = sum(d.avg_confidence_nonempty for d in passports) / len(passports) if passports else 0
    l_avg = sum(d.avg_confidence_nonempty for d in licences) / len(licences) if licences else 0

    p_pno = pct(passports, "Passport Number")
    p_exp = pct(passports, "Date of Expiry")
    p_mrz = (
        100.0
        * sum(
            1
            for d in passports
            if d.fields["MRZ line 1"].status not in ("NOT READ", "UNCLEAR")
            and d.fields["MRZ line 2"].status not in ("NOT READ", "UNCLEAR")
        )
        / len(passports)
        if passports
        else 0
    )
    l_lic = pct(licences, "Licence Number")
    l_exp = pct(licences, "Date of Expiry")

    p_ready = classify_readiness(passports, "passport")
    l_ready = classify_readiness(licences, "licence")

    manual = [
        d.filename
        for d in passports + licences
        if any(f.status in ("UNCLEAR", "NOT READ") for f in d.fields.values())
        or "split" in d.mrz_notes.lower()
    ]

    lines: list[str] = [
        "# Batch Reliability Report (Phase 3)",
        "",
        f"Generated: {ts}",
        "",
        "Engine: PaddleOCR via `test_ocr.py` (CPU, unchanged).",
        "",
        "**Note:** Without document ground truth, field ratings use `OCR VALUE PRESENT` when a plausible value appears in raw OCR. Manual verification is required before claiming correctness.",
        "",
        "---",
        "",
        "## Summary",
        "",
        "### PASSPORTS",
        f"- Total tested: {len(passports)}",
        f"- Passport number detected: {p_pno:.0f}%",
        f"- Expiry date detected: {p_exp:.0f}%",
        f"- Both MRZ lines detected: {p_mrz:.0f}%",
        f"- Average confidence (non-empty lines): {p_avg:.4f}",
        "- Main recurring errors: see per-file tables; watch MRZ spacing splits, missing Arabic visual zones, watermark/glare.",
        f"- **Readiness:** {p_ready}",
        "",
        "### UAE / DRIVING LICENCES",
        f"- Total tested: {len(licences)}",
        f"- Licence number detected: {l_lic:.0f}%",
        f"- Expiry date detected: {l_exp:.0f}%",
        f"- Average confidence (non-empty lines): {l_avg:.4f}",
        "- Main recurring errors: merged labels (`DATE: BIRTH`, `PLACE DATE`), weak/missing licence number on low-res cards, Arabic often absent while English OK.",
        f"- **Readiness:** {l_ready}",
        "",
        "---",
        "",
    ]

    def table(doc: DocResult, fields: list[str]) -> list[str]:
        out = [f"### {doc.category}: `{doc.filename}`", "", f"JSON: `{doc.json_path.name}`", ""]
        if doc.mrz_notes:
            out.append(f"MRZ notes: {doc.mrz_notes}")
            out.append("")
        out.append("| Field | Result | Confidence | Notes |")
        out.append("|-------|--------|------------|-------|")
        for f in fields:
            ev = doc.fields[f]
            note = ev.notes.replace("|", "/")
            if ev.evidence:
                note += f" Evidence: `{ev.evidence[:60]}`"
            out.append(f"| {f} | {ev.status} | {ev.confidence} | {note} |")
        out.append("")
        return out

    lines.append("## Passport details")
    lines.append("")
    pf = [
        "Full Name",
        "Passport Number",
        "Nationality",
        "Date of Birth",
        "Date of Expiry",
        "MRZ line 1",
        "MRZ line 2",
    ]
    for d in passports:
        lines.extend(table(d, pf))

    lines.append("## Licence details")
    lines.append("")
    lf = ["Name", "Licence Number", "Date of Expiry", "Date of Birth", "Nationality"]
    for d in licences:
        lines.extend(table(d, lf))

    lines.extend(
        [
            "---",
            "",
            "## Manual follow-up recommended",
            "",
        ]
    )
    if manual:
        for m in sorted(set(manual)):
            lines.append(f"- `{m}`")
    else:
        lines.append("- None flagged beyond standard manual verification.")

    return "\n".join(lines)


def main() -> int:
    if not PYTHON.exists():
        print("Missing .venv Python", file=sys.stderr)
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    passports: list[DocResult] = []
    licences: list[DocResult] = []

    for img in discover_images(PASSPORT_DIR):
        print(f"OCR passport: {img.name}...")
        jp = run_test_ocr(img)
        lines = load_lines(jp)
        doc = DocResult("passport", img.name, jp, lines)
        doc.avg_confidence_nonempty = nonempty_confidence(lines)
        eval_passport(doc)
        passports.append(doc)

    for img in discover_images(LICENCE_DIR):
        print(f"OCR licence: {img.name}...")
        jp = run_test_ocr(img)
        lines = load_lines(jp)
        doc = DocResult("licence", img.name, jp, lines)
        doc.avg_confidence_nonempty = nonempty_confidence(lines)
        eval_licence(doc)
        licences.append(doc)

    report = build_report(passports, licences)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report written: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
