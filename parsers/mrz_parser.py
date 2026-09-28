"""TD3 passport MRZ detection, normalization, parsing, and checksum validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from parsers.common import OcrElement, expand_mrz_yymmdd, normalize_spaces

MRZ_TD3_LEN = 44
MRZ_LINE1_RE = re.compile(r"^P[<A-Z0-9].+")
MRZ_CANDIDATE_CHARS = re.compile(r"^[A-Z0-9< ]+$", re.I)


@dataclass
class MrzChecks:
    document_number: bool = False
    date_of_birth: bool = False
    expiry_date: bool = False
    composite: bool = False


@dataclass
class MrzParseResult:
    line1: str | None = None
    line2: str | None = None
    valid: bool = False
    checks: MrzChecks = field(default_factory=MrzChecks)
    warnings: list[str] = field(default_factory=list)
    document_type: str | None = None
    issuing_country: str | None = None
    surname: str | None = None
    given_names: str | None = None
    passport_number: str | None = None
    nationality: str | None = None
    date_of_birth: str | None = None
    sex: str | None = None
    expiry_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "line1": self.line1,
            "line2": self.line2,
            "valid": self.valid,
            "checks": {
                "documentNumber": self.checks.document_number,
                "dateOfBirth": self.checks.date_of_birth,
                "expiryDate": self.checks.expiry_date,
                "composite": self.checks.composite,
            },
            "documentType": self.document_type,
            "issuingCountry": self.issuing_country,
            "surname": self.surname,
            "givenNames": self.given_names,
            "passportNumber": self.passport_number,
            "nationality": self.nationality,
            "dateOfBirth": self.date_of_birth,
            "sex": self.sex,
            "expiryDate": self.expiry_date,
            "warnings": self.warnings,
        }


def mrz_char_value(ch: str) -> int:
    if ch == "<":
        return 0
    if ch.isdigit():
        return int(ch)
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A") + 10
    return 0


def mrz_check_digit(data: str) -> str:
    weights = (7, 3, 1)
    total = sum(mrz_char_value(c) * weights[i % 3] for i, c in enumerate(data))
    return str(total % 10)


def normalize_mrz_line(raw: str) -> str:
    s = raw.upper().strip()
    s = s.replace(" ", "")
    s = re.sub(r"[^A-Z0-9<]", "", s)
    return s


def find_mrz_lines(elements: list[OcrElement]) -> tuple[str | None, str | None, list[str]]:
    warnings: list[str] = []
    candidates: list[str] = []
    for e in elements:
        t = normalize_mrz_line(e.text)
        if len(t) < 25:
            continue
        if not MRZ_CANDIDATE_CHARS.match(e.text.upper()):
            continue
        if t.startswith("P") or ("<<" in t and "<" in t):
            candidates.append(t)

    if not candidates:
        return None, None, warnings + ["MRZ_NOT_FOUND"]

    line1 = line2 = None
    for c in candidates:
        if MRZ_LINE1_RE.match(c) or (c.startswith("P") and "<<" in c):
            if not line1 or len(c) > len(line1):
                line1 = c
        elif re.search(r"\d{6,9}[A-Z0-9<]{10,}", c):
            if not line2 or len(c) > len(line2):
                line2 = c

    if line1 and not line2:
        for c in candidates:
            if c != line1 and len(c) >= 30:
                line2 = c
                break

    if line1 and len(line1) != MRZ_TD3_LEN:
        warnings.append("MRZ_LINE1_LENGTH_NON_STANDARD")
    if line2 and len(line2) != MRZ_TD3_LEN:
        warnings.append("MRZ_LINE2_LENGTH_NON_STANDARD")
    if not line2:
        warnings.append("MRZ_LINE2_MISSING")

    return line1, line2, warnings


def parse_mrz_td3(line1: str | None, line2: str | None) -> MrzParseResult:
    res = MrzParseResult(line1=line1, line2=line2)
    if not line1 or not line2:
        res.warnings.append("MRZ_INCOMPLETE")
        return res

    line1 = normalize_mrz_line(line1)
    line2 = normalize_mrz_line(line2)
    res.line1, res.line2 = line1, line2

    if len(line2) < MRZ_TD3_LEN:
        res.warnings.append("MRZ_LINE2_TOO_SHORT")
        return res

    line2 = line2.ljust(MRZ_TD3_LEN, "<")[:MRZ_TD3_LEN]
    line1 = line1.ljust(MRZ_TD3_LEN, "<")[:MRZ_TD3_LEN]

    res.document_type = line1[0:2].replace("<", "").strip() or "P"
    res.issuing_country = line1[2:5].replace("<", "")
    name_part = line1[5:44]
    if "<<" in name_part:
        surname, given = name_part.split("<<", 1)
    else:
        surname, given = name_part, ""
    res.surname = surname.replace("<", " ").strip() or None
    res.given_names = given.replace("<", " ").strip() or None

    doc_num_field = line2[0:9]
    doc_check = line2[9]
    res.passport_number = doc_num_field.replace("<", "").strip() or None
    if mrz_check_digit(doc_num_field) == doc_check:
        res.checks.document_number = True
    else:
        res.warnings.append("MRZ_PASSPORT_NUMBER_CHECKSUM_FAILED")

    res.nationality = line2[10:13].replace("<", "")

    dob_field = line2[13:19]
    dob_check = line2[19]
    if mrz_check_digit(dob_field) == dob_check:
        res.checks.date_of_birth = True
    else:
        res.warnings.append("MRZ_DOB_CHECKSUM_FAILED")
    dob_iso, dob_warn = expand_mrz_yymmdd(dob_field, "dob")
    res.date_of_birth = dob_iso
    res.warnings.extend(dob_warn)

    res.sex = line2[20] if line2[20] in ("M", "F", "<") else None

    exp_field = line2[21:27]
    exp_check = line2[27]
    if mrz_check_digit(exp_field) == exp_check:
        res.checks.expiry_date = True
    else:
        res.warnings.append("MRZ_EXPIRY_CHECKSUM_FAILED")
    exp_iso, exp_warn = expand_mrz_yymmdd(exp_field, "expiry")
    res.expiry_date = exp_iso
    res.warnings.extend(exp_warn)

    composite_data = line2[0:10] + line2[13:20] + line2[21:28]
    composite_check = line2[43]
    if mrz_check_digit(composite_data) == composite_check:
        res.checks.composite = True
    else:
        res.warnings.append("MRZ_COMPOSITE_CHECKSUM_FAILED")

    res.valid = (
        res.checks.document_number
        and res.checks.date_of_birth
        and res.checks.expiry_date
        and len(line2) == MRZ_TD3_LEN
    )
    return res


def parse_mrz_from_elements(elements: list[OcrElement]) -> MrzParseResult:
    l1, l2, warn = find_mrz_lines(elements)
    res = parse_mrz_td3(l1, l2)
    res.warnings = list(dict.fromkeys(warn + res.warnings))
    return res
