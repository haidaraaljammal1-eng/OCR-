import unittest

from parsers.common import OcrElement
from parsers.passport_parser import parse_passport_from_ocr


def el(i: int, text: str, conf: float = 0.99) -> OcrElement:
    return OcrElement(index=i, text=text, confidence=conf)


class TestPassportParser(unittest.TestCase):
    def test_visual_and_mrz_match(self):
        elements = [
            el(0, "Passport No A00000069"),
            el(1, "P<EGYABDELNASER<<AHMED<SAMIR<NASR<<<<<<<<<<<<"),
            el(2, "A000000697EGY6001205M1412288<<<<<<<<<<<<<<04"),
        ]
        out = parse_passport_from_ocr(elements=elements)
        self.assertEqual(out["passportNumber"], "A00000069")
        self.assertEqual(out["expiryDate"], "2014-12-28")

    def test_expiry_compact_format(self):
        elements = [
            el(0, "Date of Expiry"),
            el(1, "19NOV2024"),
        ]
        out = parse_passport_from_ocr(elements=elements)
        self.assertEqual(out["expiryDate"], "2024-11-19")

    def test_mismatch_warning(self):
        line1 = "P<EGYTEST<<A<B<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        line2 = "A000000697EGY6001205M1412288<<<<<<<<<<<<<<04"
        elements = [
            el(0, "Passport Number A11111111"),
            el(1, line1.ljust(44, "<")[:44]),
            el(2, line2),
        ]
        out = parse_passport_from_ocr(elements=elements)
        self.assertIn("PASSPORT_NUMBER_SOURCE_MISMATCH", out["warnings"])


if __name__ == "__main__":
    unittest.main()
