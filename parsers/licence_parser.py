"""UAE-oriented driving licence parser with label/value line pairing."""

from __future__ import annotations

import re
from typing import Any

from parsers.common import (
    FieldValue,
    OcrElement,
    ParseContext,
    find_label_elements,
    label_regex,
    load_ocr_json,
    nearest_value_for_label,
    parse_date_to_iso,
    quality_score,
)

NAME_LABEL = label_regex([r"^name\b", r"\bname:"])
LICENCE_LABEL = label_regex(
    [r"licen[cs]e\s*no", r"licen[cs]e\s*number", r"\bdl\s*no\b", r"driving\s*licen[cs]e\s*no"]
)
EXPIRY_LABEL = label_regex([r"expiry\s*date", r"date\s*of\s*expiry", r"\bexpiry\b", r"valid\s*(?:until|to)"])
BIRTH_LABEL = label_regex([r"date\s*of\s*birth", r"birth\s*date", r"\bdob\b", r"date:\s*birth", r"birth"])
ISSUE_LABEL = label_regex([r"issue\s*date", r"date\s*of\s*issue", r"place\s*date"])
NAT_LABEL = label_regex([r"nationality", r"جنسية"])

LICENCE_NUM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{3,14}$")


def _is_date(s: str) -> bool:
    return parse_date_to_iso(s)[0] is not None


def _is_licence_number(s: str) -> bool:
    if _is_date(s):
        return False
    if not LICENCE_NUM_RE.match(s):
        return False
    if re.fullmatch(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}", s):
        return False
    return True


def _extract_inline_name(text: str) -> str | None:
    m = re.search(r"name\s*:?\s*(.+)$", text, re.I)
    if m:
        return m.group(1).strip() or None
    m = re.search(r"^name\s+(.+)$", text, re.I)
    if m:
        return m.group(1).strip() or None
    return None


def _field_from_label(
    ctx: ParseContext,
    label_pat: re.Pattern[str],
    validator,
    kind: str = "generic",
) -> FieldValue:
    date_kind = kind if kind in ("dob", "expiry", "issue") else "generic"
    for el in find_label_elements(ctx.elements, label_pat):
        if validator is _is_date:
            val_el, strat = nearest_value_for_label(ctx.elements, el, _is_date)
            if val_el:
                iso, w = parse_date_to_iso(val_el.stripped, date_kind)
                ctx.warnings.extend(w)
                if iso:
                    return FieldValue(
                        value=iso,
                        status="CANDIDATE",
                        source=strat,
                        ocr_confidence=val_el.confidence,
                        raw_value=val_el.stripped,
                        quality_score=quality_score(val_el.confidence, strat, True),
                    )
            iso, w = parse_date_to_iso(el.text, date_kind)
            if iso:
                ctx.warnings.extend(w)
                return FieldValue(
                    value=iso,
                    status="CANDIDATE",
                    source="SAME_LINE",
                    ocr_confidence=el.confidence,
                    raw_value=el.text,
                    quality_score=quality_score(el.confidence, "SAME_LINE", True),
                )
        else:
            if kind == "name":
                inline = _extract_inline_name(el.text)
                if inline and validator(inline):
                    return FieldValue(
                        value=inline,
                        status="CANDIDATE",
                        source="SAME_LINE",
                        ocr_confidence=el.confidence,
                        quality_score=quality_score(el.confidence, "SAME_LINE", True),
                    )
            if kind == "nationality":
                m = re.search(r"nationality\s+(.+)$", el.text, re.I)
                if m and validator(m.group(1).strip()):
                    return FieldValue(
                        value=m.group(1).strip(),
                        status="CANDIDATE",
                        source="SAME_LINE",
                        ocr_confidence=el.confidence,
                        quality_score=quality_score(el.confidence, "SAME_LINE", True),
                    )
            val_el, strat = nearest_value_for_label(ctx.elements, el, validator)
            if val_el and val_el != el:
                return FieldValue(
                    value=val_el.stripped,
                    status="CANDIDATE",
                    source=strat,
                    ocr_confidence=val_el.confidence,
                    raw_value=val_el.stripped,
                    quality_score=quality_score(val_el.confidence, strat, True),
                )
            if kind == "licence":
                m = re.search(r"(?:no\.?|number)\s*:?\s*([A-Za-z0-9-]+)", el.text, re.I)
                if m and validator(m.group(1)):
                    return FieldValue(
                        value=m.group(1),
                        status="CANDIDATE",
                        source="SAME_LINE",
                        ocr_confidence=el.confidence,
                        quality_score=quality_score(el.confidence, "SAME_LINE", True),
                    )
            if kind == "name":
                inline = _extract_inline_name(el.text)
                if inline:
                    return FieldValue(
                        value=inline,
                        status="CANDIDATE",
                        source="SAME_LINE",
                        ocr_confidence=el.confidence,
                        quality_score=quality_score(el.confidence, "SAME_LINE", True),
                    )
            if kind == "nationality":
                m = re.search(r"nationality\s+(.+)$", el.text, re.I)
                if m and m.group(1).strip():
                    return FieldValue(
                        value=m.group(1).strip(),
                        status="CANDIDATE",
                        source="SAME_LINE",
                        ocr_confidence=el.confidence,
                        quality_score=quality_score(el.confidence, "SAME_LINE", True),
                    )
    return FieldValue(value=None, status="MISSING")


def parse_licence_from_ocr(
    ocr_path: str | None = None,
    elements: list[OcrElement] | None = None,
) -> dict[str, Any]:
    if elements is None:
        if not ocr_path:
            raise ValueError("ocr_path or elements required")
        _, elements = load_ocr_json(ocr_path)

    ctx = ParseContext(elements=elements)

    name_fv = _field_from_label(ctx, NAME_LABEL, lambda s: len(s) > 2, "name")
    lic_fv = _field_from_label(ctx, LICENCE_LABEL, _is_licence_number, "licence")
    exp_fv = _field_from_label(ctx, EXPIRY_LABEL, _is_date, "expiry")
    dob_fv = _field_from_label(ctx, BIRTH_LABEL, _is_date, "dob")
    issue_fv = _field_from_label(ctx, ISSUE_LABEL, _is_date, "issue")
    nat_fv = _field_from_label(ctx, NAT_LABEL, lambda s: len(s) >= 2, "nationality")

    # Issue label "PLACE DATE" on sample — if issue empty but place date has date only
    if not issue_fv.value:
        for el in find_label_elements(ctx.elements, label_regex([r"place\s*date"])):
            val_el, strat = nearest_value_for_label(ctx.elements, el, _is_date)
            if val_el:
                iso, w = parse_date_to_iso(val_el.stripped, "generic")
                ctx.warnings.extend(w + ["ISSUE_DATE_INFERRED_FROM_PLACE_DATE_LABEL"])
                if iso:
                    issue_fv = FieldValue(
                        value=iso,
                        status="REVIEW_REQUIRED",
                        source=strat,
                        ocr_confidence=val_el.confidence,
                        raw_value=val_el.stripped,
                        quality_score=quality_score(val_el.confidence, strat, False),
                    )
                    break

    field_quality = {}
    for key, fv in [
        ("fullName", name_fv),
        ("licenceNumber", lic_fv),
        ("expiryDate", exp_fv),
        ("dateOfBirth", dob_fv),
        ("nationality", nat_fv),
    ]:
        if fv.value:
            field_quality[key] = {
                "score": fv.quality_score,
                "source": fv.source,
                "ocrConfidence": fv.ocr_confidence,
                "status": fv.status,
            }

    return {
        "documentType": "DRIVING_LICENSE",
        "fullName": name_fv.value,
        "licenceNumber": lic_fv.value,
        "nationality": nat_fv.value,
        "dateOfBirth": dob_fv.value,
        "issueDate": issue_fv.value,
        "expiryDate": exp_fv.value,
        "fieldQuality": field_quality,
        "warnings": ctx.warnings,
    }
