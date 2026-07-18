import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "export_service_data.py"
SPEC = importlib.util.spec_from_file_location("export_service_data", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class NormalizeCountColumnTests(unittest.TestCase):
    def test_writes_integer_shaped_counts(self) -> None:
        frame = pd.DataFrame({"rating_count": [68997.0, 0.0, None]})
        normalized = MODULE.normalize_count_column(frame)

        self.assertEqual(str(normalized["rating_count"].dtype), "Int64")
        self.assertEqual(normalized["rating_count"].iloc[0], 68997)
        self.assertTrue(pd.isna(normalized["rating_count"].iloc[2]))

    def test_rejects_fractional_counts(self) -> None:
        frame = pd.DataFrame({"rating_count": [3.5]})

        with self.assertRaisesRegex(ValueError, "non-integer"):
            MODULE.normalize_count_column(frame)


if __name__ == "__main__":
    unittest.main()
