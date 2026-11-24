"""Hand-calculated evaluation cases and a tiny real-image experiment."""

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from lab_4.src.evaluation import (
    classification_metrics, evaluate_thresholds, roc_auc_from_distances,
)
from lab_4.src.experiments import run_experiments
from lab_4.src.wavelet_hash import WaveletHash, hamming_distance, is_similar, wavelet_hash


class EvaluationTests(unittest.TestCase):
    def test_hand_calculated_metrics(self):
        metrics = classification_metrics([1, 1, 1, 0, 0, 0, 0],
                                         [1, 1, 0, 1, 0, 0, 0])
        self.assertEqual([metrics[key] for key in ("TP", "FN", "FP", "TN")], [2, 1, 1, 3])
        for key, expected in {"accuracy": 5 / 7, "recall": 2 / 3,
                              "specificity": 3 / 4, "precision": 2 / 3}.items():
            self.assertAlmostEqual(metrics[key], expected)

    def test_undefined_ratios(self):
        negatives = classification_metrics([0, 0], [0, 0])
        self.assertTrue(math.isnan(negatives["recall"]))
        self.assertTrue(math.isnan(negatives["precision"]))
        self.assertEqual(negatives["specificity"], 1)
        positives = classification_metrics([1, 1], [1, 1])
        self.assertTrue(math.isnan(positives["specificity"]))

    def test_invalid_labels_are_not_coerced(self):
        for invalid in ([], [[0, 1]], [0, "1"], [0, 0.5], [2],
                        [float("nan")], [1 + 0j], [True]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                classification_metrics(invalid, [0])
        with self.assertRaises(ValueError):
            classification_metrics([0, 1], [0])
        with self.assertRaises(ValueError):
            classification_metrics([0], ["0"])

    def test_threshold_equality_matches_tv4(self):
        first = WaveletHash(np.zeros(64, dtype=bool))
        bits = np.zeros(64, dtype=bool)
        bits[:4] = True
        second = WaveletHash(bits)
        self.assertTrue(is_similar(first, second, threshold=4))
        metrics = evaluate_thresholds([1], [hamming_distance(first, second)], [3, 4])
        self.assertEqual([row["TP"] for row in metrics], [0, 1])
        self.assertEqual([row["FN"] for row in metrics], [1, 0])
        self.assertEqual(metrics[1]["comparison"], "<=")

    def test_invalid_distances_and_thresholds(self):
        for invalid in ([], [[1]], [-1], [0.5], [np.inf], [np.nan], [True], ["1"], [1j]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    evaluate_thresholds([1], invalid, [1])
                with self.assertRaises(ValueError):
                    evaluate_thresholds([1], [1], invalid)
                with self.assertRaises(ValueError):
                    roc_auc_from_distances([1], invalid)
        with self.assertRaises(ValueError):
            evaluate_thresholds([0, 1], [1], [1])
        with self.assertRaises(ValueError):
            roc_auc_from_distances([0, 1], [1])

    def test_roc_direction_and_ties(self):
        self.assertEqual(roc_auc_from_distances([1, 1, 0, 0], [0, 2, 4, 6])["auc"], 1)
        self.assertEqual(roc_auc_from_distances([1, 1, 0, 0], [6, 4, 2, 0])["auc"], 0)
        tied = roc_auc_from_distances([1, 0, 1, 0], [2, 2, 4, 4])
        self.assertEqual(tied, {"fpr": [0, 0.5, 1], "tpr": [0, 0.5, 1],
                                "thresholds": [-math.inf, 2, 4], "auc": 0.5})
        self.assertEqual(roc_auc_from_distances([0, 1], [0, 0])["auc"], 0.5)
        for labels in ([0, 0], [1, 1]):
            with self.assertRaisesRegex(ValueError, "both classes"):
                roc_auc_from_distances(labels, [0, 1])


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        image = np.tile(np.arange(64, dtype=np.uint8) * 4, (64, 1))
        for name, pixels in (("a.png", image), ("b.png", image), ("c.png", 255 - image)):
            self.assertTrue(cv2.imwrite(str(self.root / name), pixels))
        self.pairs = [{"image1": "a.png", "image2": "b.png", "label": "1", "notes": "copy"},
                      {"image1": "a.png", "image2": "c.png", "label": "0", "notes": "inverse"}]
        self.configs = [dict(config_id=f"haar_{mode}", wavelet="haar", level=1,
                             mode=mode, hash_size=8, image_scale=64)
                        for mode in ("median", "mean")]

    def test_real_images_two_pairs_two_configs(self):
        with patch("lab_4.src.experiments.wavelet_hash", wraps=wavelet_hash) as hashing, \
             patch("lab_4.src.experiments.hamming_distance", wraps=hamming_distance) as matching:
            result = run_experiments(self.pairs, self.root, self.configs, [0, 64])
        self.assertEqual(hashing.call_count, 6)
        self.assertEqual(matching.call_count, 4)
        self.assertEqual(len(result["pair_scores"]), 4)
        self.assertEqual(len(result["metrics_by_threshold"]), 4)
        for row in result["pair_scores"]:
            self.assertEqual(row["distance"], 0 if row["label"] == 1 else 64)
            self.assertEqual(row["score"], -row["distance"])
            self.assertEqual(row["bit_length"], 64)
            self.assertEqual(row["notes"], "copy" if row["label"] == 1 else "inverse")
        for summary in result["configuration_summary"]:
            self.assertEqual((summary["n_pairs"], summary["n_positive"], summary["n_negative"]), (2, 1, 1))
            self.assertEqual(summary["auc"], 1)

    def test_one_class_retains_metrics_without_roc(self):
        result = run_experiments(self.pairs[:1], self.root, self.configs[:1], [0])
        self.assertEqual(result["roc_curves"], {})
        self.assertTrue(math.isnan(result["configuration_summary"][0]["auc"]))
        self.assertEqual(result["metrics_by_threshold"][0]["TP"], 1)

    def test_duplicate_and_missing_data_fail_explicitly(self):
        duplicate = {**self.pairs[0], "image1": "b.png", "image2": "a.png"}
        with self.assertRaisesRegex(ValueError, "row 3.*duplicate"):
            run_experiments([*self.pairs, duplicate], self.root, self.configs, [0])
        for pair in ({"image1": "a.png", "image2": "b.png"},
                     {**self.pairs[0], "label": "0.5"}):
            with self.assertRaisesRegex(ValueError, "Pair row 1"):
                run_experiments([pair], self.root, self.configs, [0])
        with self.assertRaisesRegex(ValueError, "haar_median, image missing.png"):
            run_experiments([{**self.pairs[0], "image2": "missing.png"}], self.root, self.configs, [0])
        (self.root / "bad.png").write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "haar_median, image bad.png"):
            run_experiments([{**self.pairs[0], "image2": "bad.png"}], self.root, self.configs, [0])

    def test_invalid_config_and_hash_length(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            run_experiments(self.pairs, self.root, [self.configs[0], self.configs[0]], [0])
        with self.assertRaisesRegex(ValueError, "level"):
            run_experiments(self.pairs, self.root, [{**self.configs[0], "level": 0}], [0])
        with patch("lab_4.src.experiments.wavelet_hash", return_value=WaveletHash(np.zeros(16, dtype=bool))):
            with self.assertRaisesRegex(ValueError, "expected 64 hash bits"):
                run_experiments(self.pairs, self.root, self.configs[:1], [0])


if __name__ == "__main__":
    unittest.main()
