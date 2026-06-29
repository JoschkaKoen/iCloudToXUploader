"""Tests for response_parsing (stdlib unittest)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ic2x.utils.response_parsing import parse_json_safe, strip_json_fences  # noqa: E402


class TestResponseParsing(unittest.TestCase):
    def test_strip_fenced_json(self) -> None:
        raw = '```json\n{"a": 1}\n```'
        self.assertEqual(strip_json_fences(raw), '{"a": 1}')

    def test_parse_json_safe_simple(self) -> None:
        self.assertEqual(parse_json_safe('{"x": true}'), {"x": True})

    def test_parse_json_safe_prose_wrapped(self) -> None:
        self.assertEqual(parse_json_safe('Here:\n{"y": 2}\nThanks'), {"y": 2})

    def test_parse_json_safe_empty(self) -> None:
        self.assertIsNone(parse_json_safe(""))

    def test_parse_json_safe_invalid(self) -> None:
        self.assertIsNone(parse_json_safe("not json"))


if __name__ == "__main__":
    unittest.main()
