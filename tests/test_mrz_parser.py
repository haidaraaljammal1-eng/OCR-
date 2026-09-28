import unittest

from parsers.mrz_parser import (
    mrz_check_digit,
    normalize_mrz_line,
    parse_mrz_td3,
)


class TestMrzParser(unittest.TestCase):
    LINE1 = "P<EGYHASSANIN<<RAAFAT<FATH<MOHAMED<<<<<<<<<<"
    LINE2 = "00016021<2EGY7202021M0710301<<<<<<<<<<<<<<<6"

    def test_normalize_removes_spaces(self):
        self.assertEqual(normalize_mrz_line("P < E G Y"), "P<EGY")

    def test_full_mrz_checksums(self):
        res = parse_mrz_td3(self.LINE1, self.LINE2)
        self.assertTrue(res.checks.document_number)
        self.assertTrue(res.checks.date_of_birth)
        self.assertTrue(res.checks.expiry_date)
        self.assertEqual(res.passport_number, "00016021")
        self.assertEqual(res.date_of_birth, "1972-02-02")
        self.assertEqual(res.expiry_date, "2007-10-30")

    def test_missing_line2(self):
        res = parse_mrz_td3(self.LINE1, None)
        self.assertFalse(res.valid)
        self.assertIn("MRZ_INCOMPLETE", res.warnings)

    def test_bad_checksum(self):
        bad = self.LINE2[:9] + "0" + self.LINE2[10:]
        res = parse_mrz_td3(self.LINE1, bad)
        self.assertIn("MRZ_PASSPORT_NUMBER_CHECKSUM_FAILED", res.warnings)

    def test_check_digit_helper(self):
        self.assertEqual(mrz_check_digit("00016021<"), "2")


if __name__ == "__main__":
    unittest.main()
