import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "export_embeddings.py"
SPEC = importlib.util.spec_from_file_location("export_embeddings", MODULE_PATH)
assert SPEC and SPEC.loader
export_embeddings = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_embeddings)


class ExportEmbeddingsTest(unittest.TestCase):
    def test_embedding_columns_are_numeric_and_contiguous(self) -> None:
        frame = pd.DataFrame(columns=["movieId", "e10", "e2", "e1", "e0", "e3", "e4", "e5", "e6", "e7", "e8", "e9"])
        self.assertEqual(
            export_embeddings.embedding_columns(frame),
            [f"e{index}" for index in range(11)],
        )
        with self.assertRaises(ValueError):
            export_embeddings.embedding_columns(pd.DataFrame(columns=["movieId", "e0", "e2"]))

    def test_embedding_export_is_deterministic_and_little_endian(self) -> None:
        frame = pd.DataFrame({
            "movieId": [7, 9],
            "e0": [1.0, 0.5],
            "e1": [0.0, -1.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.bin"
            second = Path(directory) / "second.bin"
            metadata = export_embeddings.write_embeddings(first, "movieId", frame)
            export_embeddings.write_embeddings(second, "movieId", frame)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            magic, count, dim, item_size = struct.unpack("<4sIII", first.read_bytes()[:16])
            self.assertEqual((magic, count, dim, item_size), (b"EMB1", 2, 2, 2))
            self.assertEqual(metadata["count"], 2)
            self.assertEqual(metadata["dim"], 2)

    def test_export_rejects_duplicate_ids_and_non_finite_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.bin"
            duplicates = pd.DataFrame({"movieId": [1, 1], "e0": [1.0, 0.0]})
            with self.assertRaises(ValueError):
                export_embeddings.write_embeddings(path, "movieId", duplicates)
            non_finite = pd.DataFrame({"movieId": [1], "e0": [np.nan]})
            with self.assertRaises(ValueError):
                export_embeddings.write_embeddings(path, "movieId", non_finite)

    def test_history_matches_training_window_and_warm_users(self) -> None:
        ratings = pd.DataFrame({
            "userId": [10, 10, 10, 20, 20, 30],
            "movieId": [2, 9, 100, 5, 6, 7],
            "timestamp": [1, 2, 6, 3, 4, 5],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.bin"
            metadata = export_embeddings.write_history(
                path,
                ratings,
                np.array([10, 20], dtype=np.int32),
                val_fraction=1 / 6,
            )
            raw = path.read_bytes()
            self.assertEqual(raw[:4], b"HST1")
            self.assertEqual(struct.unpack("<I", raw[4:8])[0], 2)
            self.assertEqual(metadata["pairs"], 4)


if __name__ == "__main__":
    unittest.main()
