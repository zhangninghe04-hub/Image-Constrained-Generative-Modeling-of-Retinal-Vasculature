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


if __name__ == "__main__":
    unittest.main()
