"""Passport parser: MRZ-first with visual-zone fallback and cross-check."""

from __future__ import annotations

import re
from typing import Any

from parsers.common import (
    FieldValue,
    OcrElement,
    ParseContext,
    expand_mrz_yymmdd,
    find_label_elements,
    label_regex,
    load_ocr_json,
    nearest_value_for_label,
    parse_date_to_iso,
    quality_score,
)
from parsers.mrz_parser import parse_mrz_from_elements

PASSPORT_NO_RE = re.compile(r"\b([A-Z]?\d{6,9})\b")
NAME_LABEL = label_regex([r"full\s*name", r"\bname\b", r"surname", r"given\s*name", r"prénom"])
PASSPORT_LABEL = label_regex([r"passport\s*no", r"passport\s*number", r"pass\s*no", r"document\s*no"])
DOB_LABEL = label_regex([r"date\s*of\s*birth", r"\bdob\b", r"birth\s*date", r"تاريخ\s*الميلاد"])
EXP_LABEL = label_regex([r"date\s*of\s*expiry", r"\bexpiry\b", r"expiration", r"انتهاء"])
ISSUE_LABEL = label_regex([r"date\s*of\s*issue", r"\bissue\b", r"issue\s*no", r"إصدار"])
NAT_LABEL = label_regex([r"nationality", r"جنسية", r"country\s*code"])
SEX_LABEL = label_regex([r"\bsex\b", r"\bsexe\b", r"الجنس", r"النوع"])


def _is_date_like(s: str) -> bool:
    return parse_date_to_iso(s)[0] is not None


def _is_passport_candidate(s: str) -> bool:
    return bool(PASSPORT_NO_RE.fullmatch(s.replace(" ", "")))


def _pick_field(
    visual: FieldValue,
    mrz_val: str | None,
    field_name: str,
    warnings: list[str],
) -> FieldValue:
    if visual.value and mrz_val:
        v_norm = visual.value.replace("<", "").strip().upper()
        m_norm = mrz_val.replace("<", "").strip().upper()
        if v_norm == m_norm or v_norm in m_norm or m_norm in v_norm:
            visual.status = "CONFIRMED"
            visual.source = "MRZ_AND_VISUAL"
            visual.quality_score = quality_score(
                visual.ocr_confidence or 0.9, visual.source, True, True
            )
            return visual
        warnings.append(f"{field_name.upper()}_SOURCE_MISMATCH")
        visual.status = "REVIEW_REQUIRED"
        visual.quality_score = quality_score(visual.ocr_confidence or 0.5, "MISMATCH", True)
        return visual
    if mrz_val and not visual.value:
        return FieldValue(
            value=mrz_val,
            status="CANDIDATE",
            source="MRZ",
            quality_score=quality_score(0.85, "MRZ", True),
        )
    if visual.value:
        if visual.status == "MISSING":
            visual.status = "CANDIDATE"
        return visual
    return FieldValue(value=None, status="MISSING")


def _visual_passport_number(ctx: ParseContext) -> FieldValue:
    for el in find_label_elements(ctx.elements, PASSPORT_LABEL):
        m = PASSPORT_NO_RE.search(el.text)
        if m and PASSPORT_LABEL.search(el.text):
            return FieldValue(
                value=m.group(1),
                status="CANDIDATE",
                source="SAME_LINE",
                ocr_confidence=el.confidence,
                raw_value=el.text,
                quality_score=quality_score(el.confidence, "SAME_LINE", True),
            )
        val_el, strat = nearest_value_for_label(ctx.elements, el, _is_passport_candidate)
        if val_el:
            m = PASSPORT_NO_RE.search(val_el.stripped)
            if m:
                return FieldValue(
                    value=m.group(1),
                    status="CANDIDATE",
                    source=strat,
                    ocr_confidence=val_el.confidence,
                    raw_value=val_el.stripped,
                    quality_score=quality_score(val_el.confidence, strat, True),
                )
    for el in find_label_elements(ctx.elements, ISSUE_LABEL):
        val_el, strat = nearest_value_for_label(
            ctx.elements,
            el,
            lambda s: bool(re.search(r"\d", s)) and not _is_date_like(s),
        )
        if val_el:
            token = re.sub(r"[^A-Z0-9]", "", val_el.stripped.upper())
            if 6 <= len(token) <= 12:
                return FieldValue(
                    value=val_el.stripped,
                    status="REVIEW_REQUIRED",
                    source="ISSUE_NUMBER_FALLBACK",
                    ocr_confidence=val_el.confidence,
                    quality_score=quality_score(val_el.confidence, "ISSUE_NUMBER_FALLBACK", False),
                )
    for el in ctx.non_empty():
        if _is_passport_candidate(el.stripped) and el.stripped[0].isalpha():
            return FieldValue(
                value=el.stripped,
                status="CANDIDATE",
                source="HEURISTIC_LINE",
                ocr_confidence=el.confidence,
                quality_score=quality_score(el.confidence, "HEURISTIC_LINE", True),
            )
    return FieldValue(value=None, status="MISSING")


def _visual_date(ctx: ParseContext, label_pat: re.Pattern[str], kind: str) -> FieldValue:
    if kind == "dob":
        dob = _visual_dob_split(ctx)
        if dob.value:
            return dob
    for el in find_label_elements(ctx.elements, label_pat):
        val_el, strat = nearest_value_for_label(ctx.elements, el, _is_date_like)
        if val_el:
            iso, w = parse_date_to_iso(val_el.stripped, kind)
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
        iso, w = parse_date_to_iso(el.text, kind)
        if iso and ":" in el.text:
            ctx.warnings.extend(w)
            return FieldValue(
                value=iso,
                status="CANDIDATE",
                source="SAME_LINE",
                ocr_confidence=el.confidence,
                raw_value=el.text,
                quality_score=quality_score(el.confidence, "SAME_LINE", True),
            )
    return FieldValue(value=None, status="MISSING")


def _visual_name(ctx: ParseContext) -> FieldValue:
    given = surname = None
    for el in find_label_elements(ctx.elements, label_regex([r"given\s*name", r"prénom"])):
        val_el, _ = nearest_value_for_label(
            ctx.elements, el, lambda s: s.isupper() and len(s) >= 2 and "/" not in s
        )
        if val_el:
            given = val_el.stripped
    for el in find_label_elements(ctx.elements, label_regex([r"\bsurname\b", r"surname/"])):
        val_el, _ = nearest_value_for_label(
            ctx.elements, el, lambda s: s.isupper() and len(s) >= 2 and "/" not in s
        )
        if val_el:
            surname = val_el.stripped
    if given and surname:
        return FieldValue(
            value=f"{given} {surname}",
            status="CANDIDATE",
            source="GIVEN_SURNAME_LINES",
            quality_score=0.85,
        )
    for el in find_label_elements(ctx.elements, NAME_LABEL):
        if re.search(r"full\s*name|^\s*name\b|given\s*name|prénom", el.text, re.I):
            val = value_after_colon(el.text)
            if val and len(val) > 2 and not _is_date_like(val):
                return FieldValue(
                    value=val,
                    status="CANDIDATE",
                    source="SAME_LINE",
                    ocr_confidence=el.confidence,
                    quality_score=quality_score(el.confidence, "SAME_LINE", True),
                )
            val_el, strat = nearest_value_for_label(
                ctx.elements,
                el,
                lambda s: len(s) > 5 and not _is_date_like(s) and "date" not in s.lower(),
            )
            if val_el and val_el != el:
                val = val_el.stripped
                if val.lower().startswith("mr.") or val.lower().startswith("mrs."):
                    val = val.split(None, 1)[-1] if " " in val else val
                return FieldValue(
                    value=val,
                    status="CANDIDATE",
                    source=strat,
                    ocr_confidence=val_el.confidence,
                    quality_score=quality_score(val_el.confidence, strat, True),
                )
            idx = ctx.elements.index(el)
            for prev in reversed(ctx.elements[max(0, idx - 3) : idx]):
                if prev.is_empty:
                    continue
                if len(prev.stripped) > 8 and prev.stripped.isupper() and " " in prev.stripped:
                    if not any(
                        x in prev.stripped
                        for x in ("DATE", "PASSPORT", "PLACE", "BIRTH", "EXPIRY", "NAME/")
                    ):
                        return FieldValue(
                            value=prev.stripped,
                            status="CANDIDATE",
                            source="PREVIOUS_LINE",
                            ocr_confidence=prev.confidence,
                            quality_score=quality_score(prev.confidence, "PREVIOUS_LINE", True),
                        )
            iso = value_after_colon(el.text)
            if iso:
                return FieldValue(
                    value=iso,
                    status="CANDIDATE",
                    source="SAME_LINE",
                    ocr_confidence=el.confidence,
                    quality_score=quality_score(el.confidence, "SAME_LINE", True),
                )
    for el in ctx.non_empty():
        if len(el.stripped) > 12 and el.stripped.isupper() and " " in el.stripped:
            if not any(
                x in el.stripped
                for x in ("MINISTRY", "REPUBLIC", "KINGDOM", "PASSPORT", "RÉPUBLIQUE", "SYRIAN")
            ):
                return FieldValue(
                    value=el.stripped,
                    status="CANDIDATE",
                    source="HEURISTIC_NAME",
                    ocr_confidence=el.confidence,
                    quality_score=quality_score(el.confidence, "HEURISTIC_NAME", False),
                )
    return FieldValue(value=None, status="MISSING")


def _visual_dob_split(ctx: ParseContext) -> FieldValue:
    """Handle DOB split across OCR lines (e.g. day/month fragment + year)."""
    for el in find_label_elements(ctx.elements, DOB_LABEL):
        idx = ctx.elements.index(el)
        window = ctx.elements[idx : idx + 6]
        year = month = day = None
        conf = el.confidence
        for w in window:
            t = w.stripped
            if re.fullmatch(r"19\d{2}|20\d{2}", t):
                year = int(t)
                conf = w.confidence
            m = re.search(r"\b(\d{1,2})\s*([A-Za-z]{3})", t)
            if m:
                day, mon = int(m.group(1)), m.group(2).upper()[:3]
                from parsers.common import MONTHS

                month = MONTHS.get(mon)
            iso, _ = parse_date_to_iso(t, "dob")
            if iso:
                return FieldValue(
                    value=iso,
                    status="CANDIDATE",
                    source="DOB_WINDOW",
                    ocr_confidence=w.confidence,
                    quality_score=quality_score(w.confidence, "DOB_WINDOW", True),
                )
        if year and month and day:
            try:
                from datetime import date

                iso = date(year, month, day).isoformat()
                return FieldValue(
                    value=iso,
                    status="CANDIDATE",
                    source="DOB_SPLIT_LINES",
                    ocr_confidence=conf,
                    quality_score=quality_score(conf, "DOB_SPLIT_LINES", True),
                )
            except ValueError:
                pass
    return FieldValue(value=None, status="MISSING")


def value_after_colon(text: str) -> str | None:
    if ":" not in text:
        return None
    v = text.split(":", 1)[1].strip()
    return v or None


def _visual_nationality(ctx: ParseContext) -> FieldValue:
    for el in find_label_elements(ctx.elements, NAT_LABEL):
        val_el, strat = nearest_value_for_label(
            ctx.elements,
            el,
            lambda s: len(s) >= 3 and not _is_date_like(s),
        )
        if val_el and val_el != el:
            return FieldValue(
                value=val_el.stripped,
                status="CANDIDATE",
                source=strat,
                ocr_confidence=val_el.confidence,
                quality_score=quality_score(val_el.confidence, strat, True),
            )
    for el in ctx.non_empty():
        if el.stripped in ("EGY", "JOR", "SYR"):
            return FieldValue(
                value=el.stripped,
                status="CANDIDATE",
                source="COUNTRY_CODE_TOKEN",
                ocr_confidence=el.confidence,
                quality_score=quality_score(el.confidence, "COUNTRY_CODE_TOKEN", True),
            )
    return FieldValue(value=None, status="MISSING")


def _visual_sex(ctx: ParseContext) -> FieldValue:
    for el in find_label_elements(ctx.elements, SEX_LABEL):
        val_el, strat = nearest_value_for_label(
            ctx.elements, el, lambda s: s.upper() in ("M", "F", "MS", "MR")
        )
        if val_el:
            v = val_el.stripped.upper()
            if v in ("M", "F"):
                return FieldValue(value=v, status="CANDIDATE", source=strat, ocr_confidence=val_el.confidence)
    return FieldValue(value=None, status="MISSING")


def parse_passport_from_ocr(
    ocr_path: str | None = None,
    elements: list[OcrElement] | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    if elements is None:
        if not ocr_path:
            raise ValueError("ocr_path or elements required")
        _, elements = load_ocr_json(ocr_path)

    ctx = ParseContext(elements=elements)
    mrz = parse_mrz_from_elements(elements)
    warnings.extend(mrz.warnings)

    v_passport = _visual_passport_number(ctx)
    v_dob = _visual_date(ctx, DOB_LABEL, "dob")
    v_exp = _visual_date(ctx, EXP_LABEL, "expiry")
    v_name = _visual_name(ctx)
    v_nat = _visual_nationality(ctx)
    v_sex = _visual_sex(ctx)

    passport_number = _pick_field(v_passport, mrz.passport_number, "passport_number", warnings)
    dob = _pick_field(v_dob, mrz.date_of_birth, "date_of_birth", warnings)
    expiry = _pick_field(v_exp, mrz.expiry_date, "expiry_date", warnings)

    first_name = last_name = None
    if mrz.given_names:
        first_name = mrz.given_names
    if mrz.surname:
        last_name = mrz.surname
    if mrz.given_names or mrz.surname:
        mrz_full = " ".join(p for p in (mrz.given_names, mrz.surname) if p).strip()
    else:
        mrz_full = None
    full_name = v_name.value
    if mrz.valid and mrz_full:
        full_name = mrz_full
    elif not full_name and mrz_full:
        full_name = mrz_full
    elif not full_name and (first_name or last_name):
        full_name = " ".join(p for p in (first_name, last_name) if p)

    nationality = v_nat.value or mrz.nationality
    nat_status = "CANDIDATE" if nationality else "MISSING"
    if mrz.nationality and v_nat.value and v_nat.value.upper()[:3] != mrz.nationality:
        warnings.append("NATIONALITY_SOURCE_MISMATCH")

    sex = v_sex.value or (mrz.sex if mrz.sex in ("M", "F") else None)

    field_quality = {}
    for key, fv in [
        ("passportNumber", passport_number),
        ("expiryDate", expiry),
        ("dateOfBirth", dob),
        ("fullName", v_name),
    ]:
        if fv.value:
            field_quality[key] = {
                "score": fv.quality_score,
                "source": fv.source,
                "ocrConfidence": fv.ocr_confidence,
                "status": fv.status,
            }

    return {
        "documentType": "PASSPORT",
        "firstName": first_name,
        "lastName": last_name,
        "fullName": full_name,
        "passportNumber": passport_number.value,
        "nationality": nationality,
        "dateOfBirth": dob.value,
        "sex": sex,
        "expiryDate": expiry.value,
        "issuingCountry": mrz.issuing_country,
        "mrz": mrz.to_dict(),
        "fieldQuality": field_quality,
        "warnings": list(dict.fromkeys(warnings + ctx.warnings)),
    }
