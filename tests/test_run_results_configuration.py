import ast
import unittest
from pathlib import Path


def _literal_assignment(name):
    tree = ast.parse(Path("run_results.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} assignment not found")


class RunResultsConfigurationTest(unittest.TestCase):
    def test_visualization_uses_three_representative_fundus_images(self):
        self.assertEqual(_literal_assignment("SAMPLE_IMAGES"), ["01_h.jpg", "05_h.jpg", "10_h.jpg"])

    def test_batch_evaluation_uses_all_available_fundus_images(self):
        expected = [f"{i:02d}_h.jpg" for i in range(1, 16)]
        self.assertEqual(_literal_assignment("EVALUATION_IMAGES"), expected)

    def test_ablation_models_are_configured(self):
        source = Path("run_results.py").read_text(encoding="utf-8")
        for label in [
            "Density Depth Only",
            "Density Direction Only",
            "Density Survival Only",
            "Density-Aware",
        ]:
            self.assertIn(label, source)

        self.assertIn("results/ablation_summary.csv", source)

    def test_parameter_search_and_overlay_outputs_are_configured(self):
        run_results = Path("run_results.py").read_text(encoding="utf-8")
        search = Path("run_parameter_search.py").read_text(encoding="utf-8")

        self.assertIn("fig5_terminal_density_overlay.png", run_results)
        self.assertIn("matched_terminal_density_score", run_results)
        self.assertIn("density_lift_over_random", run_results)
        self.assertIn("density_angle_span", search)
        self.assertIn("density_horizon_weight", search)
        self.assertIn("horizon", run_results)
        self.assertIn("results/parameter_search.csv", search)
        self.assertIn("results/best_parameter_summary.md", search)


if __name__ == "__main__":
    unittest.main()
