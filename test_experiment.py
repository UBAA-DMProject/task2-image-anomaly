"""Checks for leakage boundaries, strict thresholds and local score mechanics."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from run_experiment import (
    ROOT, assign_roles, discover, duplicate_report, evaluate, extract_image,
    finite_scores, load_test_labels, local_scores, normal_threshold,
    outside_data, predict, supervised_threshold,
)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT / "experiment_protocol.json").read_text(encoding="utf-8"))

    def test_disjoint_reproducible_stratified_split(self):
        rows = []
        for category in self.protocol["categories"]:
            for label, count in [(0, 200), (1, 50)]:
                rows += [{"category": category, "path": f"{category}/{label}/{i}.png", "label": label, "original_split": "train"} for i in range(count)]
            rows += [{"category": category, "path": f"{category}/test/1.png", "label": None, "original_split": "test"}]
        result = assign_roles(rows, self.protocol)
        self.assertEqual(result, assign_roles(rows, self.protocol))
        for category in self.protocol["categories"]:
            for label, role, expected in [(0, "fit", 160), (0, "validation", 40), (1, "fit", 40), (1, "validation", 10)]:
                self.assertEqual(sum(r["category"] == category and r["label"] == label and r["role"] == role for r in result), expected)
        self.assertTrue(all(r["role"] == "test" for r in result if r["original_split"] == "test"))
        self.assertNotIn("role", rows[0])

    def test_strict_threshold_and_linear_quantile(self):
        np.testing.assert_array_equal(predict([0, 1, 2], 1), [0, 0, 1])
        self.assertAlmostEqual(normal_threshold([0, 10], 0.95), 9.5)
        with self.assertRaises(ValueError):
            predict([1], np.inf)

    def test_supervised_threshold_ties_and_constant_scores(self):
        self.assertEqual(supervised_threshold([0, 0, 1, 1], [0, 1, 2, 3]), 1)
        threshold = supervised_threshold([0, 1], [5, 5])
        np.testing.assert_array_equal(predict([5, 5], threshold), [1, 1])
        with self.assertRaises(ValueError):
            supervised_threshold([0, 0], [0, 1])

    def test_local_distances_against_small_hand_calculation(self):
        reference = np.stack([np.zeros((2, 2, 1)), np.full((2, 2, 1), 4)])
        query = np.ones((1, 2, 2, 1))
        score, maps = local_scores(reference, query, neighbors=1, radius=0, top_fraction=0.25)
        np.testing.assert_allclose(maps, 1)
        np.testing.assert_allclose(score, [1])
        score, maps = local_scores(reference, query, neighbors=2, radius=0, top_fraction=1)
        np.testing.assert_allclose(score, [2])

    def test_local_max_pool_and_spatial_radius(self):
        reference = np.array([[[[0], [10]], [[20], [30]]]], dtype=float)
        query = np.array([[[[0], [10]], [[20], [50]]]], dtype=float)
        scores, _ = local_scores(reference, query, neighbors=1, radius=0, top_fraction=0.25)
        np.testing.assert_allclose(scores, [20])
        swapped = reference[:, ::-1, ::-1, :]
        scores, _ = local_scores(reference, swapped, neighbors=1, radius=1, top_fraction=1)
        np.testing.assert_allclose(scores, [0])
        with self.assertRaises(ValueError):
            local_scores(reference, query, top_fraction=0)

    def test_hog_shapes_rgb_gray_and_finite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            gray = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
            path = Path(directory) / "image.png"
            Image.fromarray(gray).save(path)
            g1, l1, info = extract_image(path, self.protocol)
            self.assertEqual(g1.shape, (8100,))
            self.assertEqual(l1.shape, (15, 15, 38))
            self.assertTrue(np.isfinite(l1).all())
            Image.fromarray(np.repeat(gray[..., None], 3, axis=2)).save(path)
            g2, l2, _ = extract_image(path, self.protocol)
            np.testing.assert_allclose(g1, g2)
            np.testing.assert_allclose(l1, l2)

    def test_path_key_alignment_and_label_validation(self):
        rows = [{"path": f"{category}/test/001.png", "role": "test"} for category in ["hazelnut", "zipper"]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = {"hazelnut/test/001.png": {"original_category": "hazelnut", "label": "good"},
                       "zipper/test/001.png": {"original_category": "zipper", "label": "bad"}}
            (root / "image_anomaly_labels.json").write_text(json.dumps(content))
            labels, _ = load_test_labels(root, rows, ["hazelnut", "zipper"])
            self.assertEqual(labels["hazelnut/test/001.png"], 0)
            self.assertEqual(labels["zipper/test/001.png"], 1)
            content["zipper/test/001.png"]["original_category"] = "hazelnut"
            (root / "image_anomaly_labels.json").write_text(json.dumps(content))
            with self.assertRaises(ValueError):
                load_test_labels(root, rows, ["hazelnut", "zipper"])

    def test_metrics_orientation_and_pr_definitions(self):
        metrics = evaluate([0, 0, 1, 1], [0, 1, 2, 3], 1)
        self.assertEqual([metrics[x] for x in ["tn", "fp", "fn", "tp"]], [2, 0, 0, 2])
        self.assertEqual(metrics["auroc"], 1)
        self.assertEqual(metrics["ap"], 1)
        self.assertEqual(metrics["f1"], 1)
        with self.assertRaises(ValueError):
            evaluate([0, 1], [0, 1, 2], 1)

    def test_nonfinite_scores_and_output_protection(self):
        for values in [[], [np.nan], [np.inf], [[1, 2]]]:
            with self.assertRaises(ValueError):
                finite_scores(values)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                outside_data(Path(directory) / "results", Path(directory))

    def test_duplicate_groups_expose_split_overlap(self):
        rows = [{"path": "train/a", "role": "fit", "original_split": "train"},
                {"path": "train/b", "role": "validation", "original_split": "train"},
                {"path": "test/a", "role": "test", "original_split": "test"}]
        infos = [{"sha256": "same", "rgb_pixel_sha256": "same"}] * 3
        report = duplicate_report(rows, infos)["sha256"]
        self.assertEqual(report["extra_rows"], 2)
        self.assertEqual(report["cross_fit_validation_groups"], 1)
        self.assertEqual(report["cross_train_test_groups"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
