import unittest

from parsers.common import OcrElement
from parsers.licence_parser import parse_licence_from_ocr


def el(i: int, text: str, conf: float = 0.99, box: list[float] | None = None) -> OcrElement:
    return OcrElement(index=i, text=text, confidence=conf, box=box)


class TestLicenceParser(unittest.TestCase):
    def test_same_line_fields(self):
        elements = [
            el(0, "NAME: ALI AHMED AL-JABRI"),
            el(1, "LICENSE NO: 1234567890"),
            el(2, "EXPIRY DATE: 14/05/2033"),
            el(3, "DATE: BIRTH 01/01/1990"),
        ]
        out = parse_licence_from_ocr(elements=elements)
        self.assertEqual(out["fullName"], "ALI AHMED AL-JABRI")
        self.assertEqual(out["licenceNumber"], "1234567890")
        self.assertEqual(out["expiryDate"], "2033-05-14")
        self.assertEqual(out["dateOfBirth"], "1990-01-01")

    def test_label_next_line(self):
        elements = [
            el(0, "Expiry Date", box=[10, 100, 100, 120]),
            el(1, "", box=[10, 125, 20, 130]),
            el(2, "13/04/2023", box=[10, 130, 100, 150]),
            el(3, "License No.", box=[10, 50, 100, 70]),
            el(4, "1893918", box=[10, 75, 100, 95]),
        ]
        out = parse_licence_from_ocr(elements=elements)
        self.assertEqual(out["expiryDate"], "2023-04-13")
        self.assertEqual(out["licenceNumber"], "1893918")

    def test_missing_field_null(self):
        out = parse_licence_from_ocr(elements=[el(0, "NAME: ONLY NAME")])
        self.assertIsNone(out["licenceNumber"])
        self.assertIsNone(out["expiryDate"])

    def test_three_dates(self):
        elements = [
            el(0, "Date of Birth"),
            el(1, "04/05/1980"),
            el(2, "Issue Date"),
            el(3, "13/04/2013"),
            el(4, "Expiry Date"),
            el(5, "13/04/2023"),
        ]
        out = parse_licence_from_ocr(elements=elements)
        self.assertEqual(out["dateOfBirth"], "1980-05-04")
        self.assertEqual(out["issueDate"], "2013-04-13")
        self.assertEqual(out["expiryDate"], "2023-04-13")


if __name__ == "__main__":
    unittest.main()
