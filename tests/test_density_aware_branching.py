import unittest

import numpy as np

from src.evaluation.metrics import (
    density_lift_over_random,
    matched_terminal_density_score,
    occupied_grid_coverage,
    terminal_density_score,
)
from src.generative_models.branching_model import (
    ConstrainedTreeGenerator,
    Node,
    TreeGeneratorConfig,
)


class DensityAwareBranchingTest(unittest.TestCase):
    def test_density_guided_angle_prefers_higher_density_endpoint(self):
        density = np.zeros((10, 10), dtype=float)
        density[4, 7] = 1.0
        config = TreeGeneratorConfig(
            density_map=density,
            density_weight=1.0,
            density_depth_weight=0.0,
            density_direction_weight=1.0,
            density_survival_weight=0.0,
            branch_angle_std=0.3,
            density_candidate_angles=3,
        )
        gen = ConstrainedTreeGenerator(config)

        selected = gen._select_density_guided_angle(Node(0.0, 0.0, 0), 0.0, 0.5)

        self.assertGreater(selected, 0.0)

    def test_density_survival_probability_is_higher_in_dense_regions(self):
        density = np.zeros((10, 10), dtype=float)
        density[5, 7] = 0.05
        density[4, 7] = 1.0
        config = TreeGeneratorConfig(
            density_map=density,
            density_weight=1.0,
            density_depth_weight=0.0,
            density_direction_weight=0.0,
            density_survival_weight=1.0,
        )
        gen = ConstrainedTreeGenerator(config)

        sparse = gen._density_survival_probability(0.45, -0.1, depth=6)
        dense = gen._density_survival_probability(0.45, 0.1, depth=6)

        self.assertGreater(dense, sparse)

    def test_terminal_density_score_samples_target_density(self):
        density = np.zeros((10, 10), dtype=float)
        density[5, 5] = 1.0

        score = terminal_density_score([(0.0, 0.0)], density)

        self.assertEqual(score, 1.0)

    def test_occupied_grid_coverage_increases_with_more_cells(self):
        one_cell = occupied_grid_coverage([(0.0, 0.0)], grid_size=10)
        two_cells = occupied_grid_coverage([(0.0, 0.0), (0.5, 0.0)], grid_size=10)

        self.assertGreater(two_cells, one_cell)

    def test_matched_terminal_density_score_uses_requested_sample_size(self):
        density = np.zeros((10, 10), dtype=float)
        density[5, 5] = 1.0
        density[5, 6] = 0.5
        points = [(0.0, 0.0), (0.2, 0.0)]

        score = matched_terminal_density_score(
            points, density, sample_size=1, n_repeats=4, random_seed=1
        )

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_density_lift_over_random_returns_finite_value(self):
        density = np.zeros((10, 10), dtype=float)
        density[5, 5] = 1.0

        lift = density_lift_over_random(
            [(0.0, 0.0)], density, sample_size=1, n_repeats=5, random_seed=1
        )

        self.assertTrue(np.isfinite(lift))


if __name__ == "__main__":
    unittest.main()
