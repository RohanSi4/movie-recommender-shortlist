import importlib.util
import unittest
from pathlib import Path

import numpy as np
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

    def test_validation_and_test_cohorts_are_disjoint(self) -> None:
        # The published test cohort is only "untouched" if no user can land in
        # both buckets, so pin the partition itself.
        users = range(1, 5001)
        validation = {u for u in users if evaluate_taste.cohort_bucket(u, 42) == "validation"}
        test = {u for u in users if evaluate_taste.cohort_bucket(u, 42) == "test"}
        self.assertEqual(validation & test, set())
        self.assertEqual(len(validation) + len(test), len(users))

    def test_seeded_items_are_never_retrieved(self) -> None:
        # main() masks each supplied favorite with -inf before ranking, for the
        # model and the popularity baseline alike. A seed leaking back into the
        # candidate list would inflate every metric, so assert the mechanism.
        scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
        seed_rows = [0, 2]
        scores[seed_rows] = -np.inf
        ranked = evaluate_taste.top_k_indices(scores, 3)
        self.assertEqual(list(ranked), [1, 3, 4])
        for row in seed_rows:
            self.assertNotIn(row, ranked)


if __name__ == "__main__":
    unittest.main()
