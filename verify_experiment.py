"""Independent checks of published predictions, saved models and data integrity."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from run_experiment import ROOT, extract_image, save_json, score_model, sha256


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def independently_evaluate(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores)
    positive, negative = scores[labels == 1], scores[labels == 0]
    rank = np.mean((positive[:, None] > negative).astype(float) + 0.5 * (positive[:, None] == negative))
    order = np.argsort(-scores, kind="stable")
    sorted_labels, sorted_scores = labels[order], scores[order]
    endpoints = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores)-1]
    tp = np.cumsum(sorted_labels)[endpoints]
    precisions, recalls = tp / (endpoints+1), tp / len(positive)
    ap = np.sum(np.diff(np.r_[0, recalls]) * precisions)
    auprc = np.trapezoid(np.r_[1, precisions], np.r_[0, recalls])
    predictions = scores > threshold
    tn = int(np.sum((labels == 0) & ~predictions))
    fp = int(np.sum((labels == 0) & predictions))
    fn = int(np.sum((labels == 1) & ~predictions))
    true_pos = int(np.sum((labels == 1) & predictions))
    precision = true_pos / (true_pos+fp) if true_pos+fp else 0
    recall = true_pos / (true_pos+fn)
    f1 = 2*true_pos / (2*true_pos+fp+fn) if 2*true_pos+fp+fn else 0
    return {"auroc": rank, "ap": ap, "auprc_trapezoid": auprc,
            "precision": precision, "recall": recall, "f1": f1,
            "tn": tn, "fp": fp, "fn": fn, "tp": true_pos}


def run(output):
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    frozen = json.loads((output / "frozen_models.json").read_text(encoding="utf-8"))
    events = json.loads((output / "events.json").read_text(encoding="utf-8"))
    root = Path(provenance["data_root"])
    inventory = csv_rows(output / "inventory.csv")
    predictions = csv_rows(output / "predictions.csv")
    validation = csv_rows(output / "validation_predictions.csv")
    grid = csv_rows(output / "svm_validation_search.csv")
    names = [e["event"] for e in events]
    expected = ["protocol_registered", "all_models_and_thresholds_frozen", "unlabelled_test_predictions_saved", "test_labels_loaded_for_evaluation", "evaluation_and_plots_complete"]
    assert names == expected, names
    assert np.all(np.diff([e["elapsed_seconds"] for e in events]) > 0)
    assert events[1]["manifest_sha256"] == sha256(output / "frozen_models.json")
    assert events[3]["labels_sha256"] == sha256(root / "image_anomaly_labels.json")
    assert provenance["protocol_sha256"] == sha256(ROOT / "experiment_protocol.json")
    assert protocol == json.loads((ROOT / "experiment_protocol.json").read_text(encoding="utf-8"))
    assert all(sha256(root/r["path"]) == r["sha256"] for r in inventory)
    assert all(sha256(ROOT/name) == value for name, value in provenance["code_sha256"].items())
    test_rows = [r for r in inventory if r["role"] == "test"]
    assert len(test_rows) == 102 and len(predictions) == 306
    with ThreadPoolExecutor(max_workers=4) as pool:
        features = list(pool.map(lambda r: extract_image(root/r["path"], protocol), test_rows))
    global_features, local_features = np.stack([x[0] for x in features]), np.stack([x[1] for x in features])
    max_error, checked = 0.0, []
    with np.load(ROOT/"cache"/(summary["data_checks"]["features"]["key"]+".npz"), allow_pickle=False) as cache:
        cached_global = cache["global_features"]
    with threadpool_limits(limits=4):
        for metric in summary["metrics"]:
            category, method = metric["category"], metric["method"]
            group = [r for r in predictions if r["category"] == category and r["method"] == method]
            assert len({r["path"] for r in group}) == len(group)
            info = frozen[category][method]
            model_path = output/"models"/f"{category}_{method}.joblib"
            assert sha256(model_path) == info["model_sha256"]
            model = joblib.load(model_path)
            fit = [r for r in inventory if r["category"] == category and r["role"] == "fit" and
                   (method == "hog_linear_svm" or r["label"] == "0")]
            assert model["fit_paths"] == [r["path"] for r in fit]
            assert not set(model["fit_paths"]) & {r["path"] for r in inventory if r["role"] != "fit"}
            cal = [r for r in validation if r["category"] == category and r["method"] == method]
            if method != "hog_linear_svm":
                threshold = np.quantile([float(r["score"]) for r in cal if r["true_label"] == "0"], 0.95, method="linear")
                assert threshold == info["threshold"]
            else:
                candidates = [r for r in grid if r["category"] == category]
                best = sorted(candidates, key=lambda r: (-float(r["validation_ap"]), float(r["C"])))[0]
                assert float(best["C"]) == model["selected_C"] == info["selected_C"]
                positions = [i for i, r in enumerate(inventory) if r["category"] == category and r["role"] == "fit"]
                np.testing.assert_allclose(model["estimator"][0].mean_, cached_global[positions].astype(float).mean(axis=0), atol=1e-12, rtol=1e-12)
            indices = [i for i, r in enumerate(test_rows) if r["category"] == category]
            recalculated, _ = score_model(model, global_features[indices], local_features[indices])
            assert [test_rows[i]["path"] for i in indices] == [r["path"] for r in group]
            saved = np.array([float(r["score"]) for r in group])
            np.testing.assert_allclose(recalculated, saved, atol=1e-7, rtol=1e-7)
            max_error = max(max_error, float(np.max(np.abs(recalculated-saved))))
            np.testing.assert_array_equal(saved > model["threshold"], [int(r["predicted_label"]) for r in group])
            independent = independently_evaluate([int(r["true_label"]) for r in group], saved, model["threshold"])
            for name, value in independent.items():
                np.testing.assert_allclose(metric[name], value, rtol=1e-12, atol=1e-12)
            checked.append(f"{category}/{method}")
    result = {"status": "passed", "checked_models": checked, "verified_predictions": len(predictions),
        "redecoded_test_images": len(test_rows), "max_reloaded_score_error": max_error,
        "original_image_hashes_verified": len(inventory), "phase_order_verified": True,
        "model_hashes_verified": True, "source_code_hashes_verified": True,
        "svm_scaler_fit_scope_verified": True, "normal_quantiles_verified": True,
        "independent_metric_calculation": "pairwise AUROC; grouped-ranking AP; count-based classification; trapezoidal PR"}
    save_json(output / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    run(parser.parse_args().result_dir.resolve())
