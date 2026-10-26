import json
import unittest
from pathlib import Path

import numpy as np

from src.algorithms import box_mean, evaluate, segment, suppress_impulses


class NumericalChecks(unittest.TestCase):
    def test_impulse_filter_preserves_thin_stroke(self):
        image = np.full((15, 15), 240, dtype=np.uint8)
        image[3:12, 7] = 70
        image[2, 2] = 0
        result = suppress_impulses(image)
        np.testing.assert_array_equal(result[3:12, 7], image[3:12, 7])
        self.assertEqual(result[2, 2], 240)

    def test_integral_against_explicit_neighborhoods(self):
        image = np.random.default_rng(13).integers(0, 256, (8, 9)).astype(float)
        for window in (3, 5, 11):
            padded = np.pad(image, window // 2, mode="reflect")
            expected = np.empty_like(image)
            for row in range(image.shape[0]):
                for column in range(image.shape[1]):
                    expected[row, column] = padded[row:row + window, column:column + window].mean()
            np.testing.assert_allclose(box_mean(image, window), expected, atol=1e-10)

    def test_foreground_metrics_by_hand(self):
        truth = np.array([[True, True], [False, False]])
        prediction = np.array([[True, False], [True, False]])
        metrics = evaluate(prediction, truth)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 1))
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["iou"], 1 / 3)

    def test_global_threshold_separates_two_levels(self):
        image = np.full((20, 20), 220, dtype=np.uint8)
        image[5:15, 7:13] = 30
        np.testing.assert_array_equal(segment(image, "otsu"), image == 30)

    def test_data_split_has_no_shared_layout(self):
        manifest = json.loads((Path(__file__).parent / "data/manifest.json").read_text(encoding="utf-8"))
        validation = {row["layout_seed"] for row in manifest if row["split"] == "validation"}
        test = {row["layout_seed"] for row in manifest if row["split"] == "test"}
        self.assertFalse(validation & test)
        self.assertEqual(sum(row["split"] == "test" for row in manifest), 48)


if __name__ == "__main__":
    unittest.main(verbosity=2)
