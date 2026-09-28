"""Deterministic document parsers over PaddleOCR JSON (Phase 4)."""

from parsers.licence_parser import parse_licence_from_ocr
from parsers.passport_parser import parse_passport_from_ocr

__all__ = ["parse_passport_from_ocr", "parse_licence_from_ocr"]
