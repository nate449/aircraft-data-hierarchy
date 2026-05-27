"""Tests for Activity and Behavior validate_non_empty validators."""

import unittest
from pydantic import ValidationError

from aircraft_data_hierarchy.behavior import Activity, Behavior


WHITESPACE_VARIANTS = ["", " ", "   ", "\t", "\n", "\r\n", " \t\n\r "]
VALIDATED_FIELDS = ["name", "description"]


class TestActivity(unittest.TestCase):

    def test_default_construction(self):
        a = Activity()
        self.assertIsNone(a.name)
        self.assertIsNone(a.description)

    def test_none_accepted(self):
        for field in VALIDATED_FIELDS:
            a = Activity(**{field: None})
            self.assertIsNone(getattr(a, field))

    def test_rejects_whitespace(self):
        for field in VALIDATED_FIELDS:
            for bad in WHITESPACE_VARIANTS:
                with self.assertRaises(
                    ValidationError,
                    msg=f"Activity.{field} should reject {bad!r}",
                ):
                    Activity(**{field: bad})

    def test_preserves_whitespace_padding(self):
        # NOTE: Activity's validate_non_empty does not strip; CommonBaseModel.strip_strings
        # does not apply because the subclass validator overrides the wildcard for these fields.
        for field in VALIDATED_FIELDS:
            a = Activity(**{field: "  valid  "})
            self.assertEqual(getattr(a, field), "  valid  ")

    def test_preserves_tabs_newlines(self):
        # NOTE: stripping is not applied — see test_preserves_whitespace_padding.
        for field in VALIDATED_FIELDS:
            a = Activity(**{field: "\thello\n"})
            self.assertEqual(getattr(a, field), "\thello\n")

    def test_accepts_minimal(self):
        for field in VALIDATED_FIELDS:
            a = Activity(**{field: "a"})
            self.assertEqual(getattr(a, field), "a")


class TestBehavior(unittest.TestCase):

    def test_default_construction(self):
        b = Behavior()
        self.assertIsNone(b.name)
        self.assertIsNone(b.description)

    def test_none_accepted(self):
        for field in VALIDATED_FIELDS:
            b = Behavior(**{field: None})
            self.assertIsNone(getattr(b, field))

    def test_rejects_whitespace(self):
        for field in VALIDATED_FIELDS:
            for bad in WHITESPACE_VARIANTS:
                with self.assertRaises(
                    ValidationError,
                    msg=f"Behavior.{field} should reject {bad!r}",
                ):
                    Behavior(**{field: bad})

    def test_preserves_whitespace_padding(self):
        # NOTE: Behavior's validate_non_empty (mode="before") does not strip;
        # CommonBaseModel.strip_strings does not apply because the subclass validator
        # overrides the wildcard for these fields.
        for field in VALIDATED_FIELDS:
            b = Behavior(**{field: "  valid  "})
            self.assertEqual(getattr(b, field), "  valid  ")

    def test_preserves_tabs_newlines(self):
        # NOTE: stripping is not applied — see test_preserves_whitespace_padding.
        for field in VALIDATED_FIELDS:
            b = Behavior(**{field: "\thello\n"})
            self.assertEqual(getattr(b, field), "\thello\n")

    def test_accepts_minimal(self):
        for field in VALIDATED_FIELDS:
            b = Behavior(**{field: "a"})
            self.assertEqual(getattr(b, field), "a")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
