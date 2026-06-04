from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from common import derive_gom_batch_size, extract_elite_expressions, parse_generation_rows  # noqa: E402


class ExperimentCommonTests(unittest.TestCase):
    def test_machine_generation_records_take_precedence(self) -> None:
        stdout = """
 ~ macro generation: 1, curr. best fit: 2.0
Generation record: generation=1; elapsed_sec=0.25; best_train_fitness=2.0; num_evaluations=10; avg_actual_gpu_batch_size=7.5; best_expression=(x_0 + 1.0)
"""
        rows = parse_generation_rows(stdout)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["generation"], 1)
        self.assertEqual(rows[0]["elapsed_sec"], 0.25)
        self.assertEqual(rows[0]["num_evaluations"], 10)
        self.assertEqual(rows[0]["avg_actual_gpu_batch_size"], 7.5)
        self.assertEqual(rows[0]["best_expression"], "(x_0 + 1.0)")

    def test_extract_elite_expressions(self) -> None:
        stdout = """
All elites found:
3 1.5:(x_0 + 1.0)
1 2.0:x_0

Best w.r.t. complexity for chosen importance:
(x_0 + 1.0)
"""
        self.assertEqual(extract_elite_expressions(stdout), ["(x_0 + 1.0)", "x_0"])

    def test_derive_gom_batch_size_excludes_initial_population_launch(self) -> None:
        evaluations = 4096 + 10 * 32
        raw_average = evaluations / 11
        self.assertEqual(derive_gom_batch_size("gpu_batch_gom", raw_average, evaluations, 4096), 32.0)
        self.assertEqual(derive_gom_batch_size("gpu_exact_gom", 4096.0, evaluations, 4096), 1.0)
        self.assertTrue(math.isnan(derive_gom_batch_size("cpu_original", "", 0, 4096)))


if __name__ == "__main__":
    unittest.main()
