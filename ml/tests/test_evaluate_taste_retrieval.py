import importlib.util
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_taste_retrieval.py"
SPEC = importlib.util.spec_from_file_location("evaluate_taste_retrieval", MODULE_PATH)
assert SPEC and SPEC.loader
evaluate_taste = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_taste)


class TasteEvaluationTest(unittest.TestCase):
    def test_ranking_metrics_use_later_positives_as_truth(self) -> None:
        metrics = evaluate_taste.ranking_metrics(
            list(range(1, 101)), {2, 101}
        )
        self.assertEqual(metrics["hit_rate@10"], 1.0)
        self.assertEqual(metrics["precision@10"], 0.1)
        self.assertEqual(metrics["recall@10"], 0.5)
        self.assertEqual(metrics["mrr@10"], 0.5)

    def test_cold_user_selection_excludes_training_users(self) -> None:
        train = pd.DataFrame({"userId": [1], "movieId": [10]})
        future = pd.DataFrame({
            "userId": [1, 1, 2, 2, 2],
            "movieId": [11, 12, 20, 21, 22],
            "timestamp": [1, 2, 1, 2, 3],
        })
        users, histories, counts = evaluate_taste.select_cold_users(
            train,
            future,
            min_positives=3,
            cohort="all",
            split_seed=42,
            max_users=0,
        )
        self.assertEqual(users, [2])
        self.assertEqual(histories[2], [20, 21, 22])
        self.assertEqual(counts["eligible_all"], 1)

    def test_cohort_assignment_is_deterministic(self) -> None:
        first = evaluate_taste.cohort_bucket(123, 42)
        second = evaluate_taste.cohort_bucket(123, 42)
        self.assertEqual(first, second)
        self.assertIn(first, {"validation", "test"})


if __name__ == "__main__":
    unittest.main()
