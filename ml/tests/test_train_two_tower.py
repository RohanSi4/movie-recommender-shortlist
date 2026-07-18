import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "train_two_tower.py"
SPEC = importlib.util.spec_from_file_location("train_two_tower", MODULE_PATH)
assert SPEC and SPEC.loader
train_two_tower = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_two_tower)


class RetrievalTrainingTest(unittest.TestCase):
    def test_sampled_softmax_uses_unique_candidate_labels(self) -> None:
        queries = torch.eye(2)
        candidates = torch.eye(2)
        labels = torch.tensor([0, 1])
        loss = train_two_tower.sampled_softmax_loss(
            queries, candidates, labels, temperature=0.05
        )
        self.assertLess(float(loss), 1e-6)

    def test_user_balanced_q_matches_sampler(self) -> None:
        users = np.array([0, 0, 1], dtype=np.int64)
        items = np.array([0, 1, 1], dtype=np.int64)
        offsets, values = train_two_tower.build_positive_index(users, items, 2)
        balanced = train_two_tower.item_sampling_probabilities(
            users, items, offsets, n_items=2, strategy="user-balanced"
        )
        interaction = train_two_tower.item_sampling_probabilities(
            users, items, offsets, n_items=2, strategy="interaction"
        )
        np.testing.assert_allclose(balanced, [0.25, 0.75])
        np.testing.assert_allclose(interaction, [1 / 3, 2 / 3])
        self.assertEqual(values.tolist(), [0, 1, 1])

    def test_balanced_batches_have_unique_users_and_valid_targets(self) -> None:
        offsets = np.array([0, 3, 5], dtype=np.int64)
        values = np.array([1, 2, 3, 8, 9], dtype=np.int64)
        batches = train_two_tower.user_balanced_batches(
            offsets, values, batch_size=2, steps=3, rng=np.random.default_rng(7)
        )
        allowed = {0: {1, 2, 3}, 1: {8, 9}}
        for users, targets in batches:
            self.assertEqual(len(users.unique()), len(users))
            for user, target in zip(users.tolist(), targets.tolist()):
                self.assertIn(target, allowed[user])

    def test_taste_contexts_never_include_the_target(self) -> None:
        offsets = np.array([0, 3, 5], dtype=np.int64)
        values = np.array([1, 2, 3, 8, 9], dtype=np.int64)
        users = np.array([0, 1], dtype=np.int64)
        targets = np.array([2, 8], dtype=np.int64)
        contexts, mask = train_two_tower.sample_taste_contexts(
            users,
            targets,
            offsets,
            values,
            max_seeds=5,
            rng=np.random.default_rng(11),
        )
        for row, target in enumerate(targets):
            selected = contexts[row][mask[row]]
            self.assertGreaterEqual(len(selected), 1)
            self.assertLessEqual(len(selected), min(5, offsets[row + 1] - offsets[row] - 1))
            self.assertNotIn(target, selected)


if __name__ == "__main__":
    unittest.main()
