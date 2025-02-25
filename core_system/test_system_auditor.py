import os
import sys
import unittest
from unittest.mock import mock_open, patch

from scripts.comprehensive_system_audit import SutazAiSystemAuditor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestSutazAiSystemAuditor(unittest.TestCase):

    def setUp(self):
        self.auditor = SutazAiSystemAuditor()

    def test_detect_file_encoding(self):
        with patch("builtins.open", mock_open(read_data=b"test data")):
            encoding = self.auditor._detect_file_encoding("test.txt")
            self.assertEqual(encoding, "ascii")

    def test_is_excluded_path(self):
        excluded_path = "/path/to/.venv/file.py"
        included_path = "/path/to/src/file.py"

        self.assertTrue(self.auditor._is_excluded_path(excluded_path))
        self.assertFalse(self.auditor._is_excluded_path(included_path))

    def test_rename_references(self):
        content = "This is a SutazAi test with sutazai references."
        expected_content = "This is a SutazAi test with sutazai references."

        renamed_content, changes = self.auditor.rename_references(content)

        self.assertEqual(renamed_content, expected_content)
        self.assertEqual(changes, 2)

    def test_validate_syntax(self):
        valid_code = "def test_func():\n    print('Hello, World!')"
        invalid_code = "def test_func()\n    print('Hello, World!')"

        with patch("builtins.open", mock_open(read_data=valid_code)):
            errors = self.auditor.validate_syntax("valid.py")
            self.assertEqual(len(errors), 0)

        with patch("builtins.open", mock_open(read_data=invalid_code)):
            errors = self.auditor.validate_syntax("invalid.py")
            self.assertEqual(len(errors), 1)

    def test_optimize_performance(self):
        code_with_list_comp = "result = [x*2 for x in range(10)]"
        code_with_repeated_comp = "def calc():\n    return expensive_func()"

        with patch("builtins.open", mock_open(read_data=code_with_list_comp)):
            suggestions = self.auditor.optimize_performance(
                "test_list_comp.py"
            )
            self.assertEqual(len(suggestions), 1)
            self.assertEqual(suggestions[0]["type"], "generator_expression")

        with patch(
            "builtins.open", mock_open(read_data=code_with_repeated_comp)
        ):
            suggestions = self.auditor.optimize_performance(
                "test_repeated_comp.py"
            )
            self.assertEqual(len(suggestions), 1)
            self.assertEqual(suggestions[0]["type"], "memoization")

    @patch(
        "scripts.comprehensive_system_audit.SutazAiSystemAuditor.process_file"
    )
    def test_run_comprehensive_audit(self, mock_process_file):
        file_structure = {
            "src": {
                "main.py": 'print("Hello, World!")',
                "utils": {
                    "__init__.py": "",
                    "helper.py": 'def greet(name):\n    print(f"Hello, {name}!")',
                },
            },
            "tests": {"test_main.py": "def test_main():\n    assert True"},
        }

        with patch("os.walk") as mock_walk:
            mock_walk.return_value = [
                ("/root", ("src", "tests"), ()),
                ("/root/src", ("utils",), ("main.py",)),
                ("/root/src/utils", (), ("__init__.py", "helper.py")),
                ("/root/tests", (), ("test_main.py",)),
            ]

            with patch("builtins.open", mock_open(read_data="file content")):
                self.auditor.run_comprehensive_audit()

        self.assertEqual(mock_process_file.call_count, 3)


if __name__ == "__main__":
    unittest.main()
