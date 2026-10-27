"""Reproducible image-level anomaly detection for the required course task 2.

The protocol is committed to disk before fitting. Test labels are loaded only
after models, thresholds and unlabelled test predictions have been saved.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import hashlib
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import joblib
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from skimage.feature import hog
from skimage.util import view_as_windows
from sklearn.metrics import (
    auc, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
METHODS = ("global_hog_knn", "local_hog_knn", "hog_linear_svm")


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, rows, fields=None):
    import csv
    rows = list(rows)
    if fields is None:
        if not rows:
            raise ValueError("Explicit field names are required for an empty CSV")
        fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outside_data(path, data_root):
    path, data_root = Path(path).resolve(), Path(data_root).resolve()
    if path == data_root or data_root in path.parents:
        raise ValueError("Outputs and caches must be outside the original dataset")
    return path


def discover(data_root, protocol):
    """Discover paths and training labels only; never open the test-label JSON."""
    rows = []
    for category in protocol["categories"]:
        for folder, label, expected in [("train/good", 0, 200), ("train/bad", 1, 50), ("test", None, None)]:
            paths = sorted((data_root / category / folder).glob("*.png"))
            if not paths or (expected is not None and len(paths) != expected):
                raise ValueError(f"Unexpected image count in {category}/{folder}: {len(paths)}")
            for path in paths:
                rows.append({"path": path.relative_to(data_root).as_posix(), "category": category,
                             "original_split": "test" if folder == "test" else "train", "label": label})
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate relative image paths")
    return rows


def assign_roles(rows, protocol):
    result = [dict(row, role="test" if row["original_split"] == "test" else "fit") for row in rows]
    for category in protocol["categories"]:
        for label in (0, 1):
            positions = [i for i, row in enumerate(result) if row["category"] == category and row["label"] == label]
            order = np.random.default_rng(protocol["seed"]).permutation(positions)
            n_val = math.ceil(len(order) * protocol["validation_fraction"])
            if not 0 < n_val < len(order):
                raise ValueError("Each training class needs fit and validation samples")
            for index in order[:n_val]:
                result[int(index)]["role"] = "validation"
    return result


def extract_image(path, protocol):
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as source:
        source.load()
        mode, size = source.mode, source.size
        rgb = source.convert("RGB")
        pixel_hash = hashlib.sha256(str(size).encode("ascii") + rgb.tobytes()).hexdigest()
        image = source.convert("L").resize((protocol["image_size"],) * 2, Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    settings = protocol["hog"]
    blocks = hog(pixels, orientations=settings["orientations"],
                 pixels_per_cell=tuple(settings["pixels_per_cell"]),
                 cells_per_block=tuple(settings["cells_per_block"]),
                 block_norm=settings["block_norm"], transform_sqrt=settings["transform_sqrt"],
                 feature_vector=False)
    block_vectors = blocks.reshape(*blocks.shape[:2], -1).astype(np.float32)
    patch_size = tuple(a * b for a, b in zip(settings["pixels_per_cell"], settings["cells_per_block"]))
    windows = view_as_windows(pixels, patch_size, step=tuple(settings["pixels_per_cell"]))
    local = np.concatenate([block_vectors, windows.mean(axis=(-2, -1))[..., None],
                            windows.std(axis=(-2, -1))[..., None]], axis=-1).astype(np.float32)
    if not np.isfinite(local).all():
        raise ValueError(f"Nonfinite descriptor: {path}")
    info = {"sha256": hashlib.sha256(raw).hexdigest(), "rgb_pixel_sha256": pixel_hash,
            "width": size[0], "height": size[1], "mode": mode, "bytes": len(raw)}
    return block_vectors.reshape(-1), local, info


def load_features(data_root, rows, protocol, cache_dir, workers):
    image_hashes = [sha256(data_root / row["path"]) for row in rows]
    feature_config = {key: protocol[key] for key in ("image_size", "color_mode", "interpolation", "hog")}
    import inspect
    feature_config["extractor_sha256"] = hashlib.sha256(inspect.getsource(extract_image).encode()).hexdigest()
    feature_config["packages"] = {name: version(name) for name in ("numpy", "scikit-image", "Pillow")}
    key = hashlib.sha256(json.dumps({"config": feature_config, "paths": [r["path"] for r in rows],
                                    "hashes": image_hashes}, sort_keys=True).encode()).hexdigest()
    cache = cache_dir / key
    archive, metadata = cache.with_suffix(".npz"), cache.with_suffix(".json")
    if archive.exists() and metadata.exists():
        infos = json.loads(metadata.read_text(encoding="utf-8"))
        with np.load(archive, allow_pickle=False) as saved:
            global_features, local_features = saved["global_features"], saved["local_features"]
        if len(infos) != len(rows) or [info["sha256"] for info in infos] != image_hashes:
            raise ValueError("Feature cache does not match the images")
        cached = True
    else:
        print(f"Extracting fixed HOG descriptors from {len(rows)} images...", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            extracted = list(pool.map(lambda row: extract_image(data_root / row["path"], protocol), rows))
        global_features = np.stack([x[0] for x in extracted])
        local_features = np.stack([x[1] for x in extracted])
        infos = [x[2] for x in extracted]
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(archive, global_features=global_features, local_features=local_features)
        save_json(metadata, infos)
        cached = False
    if global_features.shape[0] != len(rows) or local_features.shape[0] != len(rows):
        raise ValueError("Feature and path counts differ")
    if not np.isfinite(global_features).all() or not np.isfinite(local_features).all():
        raise ValueError("Nonfinite cached features")
    return global_features, local_features, infos, {"key": key, "reused": cached,
        "global_shape": list(global_features.shape), "local_shape": list(local_features.shape)}


def duplicate_report(rows, infos):
    reports = {}
    for field in ("sha256", "rgb_pixel_sha256"):
        groups = defaultdict(list)
        for row, info in zip(rows, infos):
            groups[info[field]].append(row)
        repeated = [group for group in groups.values() if len(group) > 1]
        reports[field] = {"duplicate_groups": len(repeated),
                         "extra_rows": sum(len(group) - 1 for group in repeated),
                         "cross_fit_validation_groups": sum({"fit", "validation"} <= {r["role"] for r in g} for g in repeated),
                         "cross_train_test_groups": sum({"train", "test"} <= {r["original_split"] for r in g} for g in repeated),
                         "groups": [[r["path"] for r in group] for group in repeated]}
    return reports


def select_indices(rows, category, role, label=None):
    return np.asarray([i for i, row in enumerate(rows) if row["category"] == category and row["role"] == role
                       and (label is None or row["label"] == label)], dtype=int)


def finite_scores(values):
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected a nonempty, finite, one-dimensional score vector")
    return scores


def predict(scores, threshold):
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    return (finite_scores(scores) > threshold).astype(np.int64)


def normal_threshold(scores, quantile):
    if not 0 < quantile < 1:
        raise ValueError("Quantile must be strictly between zero and one")
    return float(np.quantile(finite_scores(scores), quantile, method="linear"))


def supervised_threshold(labels, scores):
    labels, scores = np.asarray(labels), finite_scores(scores)
    if set(np.unique(labels)) != {0, 1} or labels.shape != scores.shape:
        raise ValueError("Threshold calibration needs both validation classes")
    thresholds = np.r_[np.nextafter(scores.min(), -np.inf), np.unique(scores)]
    best = None
    for threshold in thresholds:
        predictions = predict(scores, threshold)
        f1 = float(f1_score(labels, predictions, zero_division=0))
        recall = float(recall_score(labels, predictions, zero_division=0))
        precision = float(precision_score(labels, predictions, zero_division=0))
        key = (f1, recall, precision, float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def evaluate(labels, scores, threshold):
    labels, scores = np.asarray(labels), finite_scores(scores)
    if labels.ndim != 1 or labels.shape != scores.shape or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation requires aligned binary labels containing both classes")
    predictions = predict(scores, threshold)
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {"auroc": float(roc_auc_score(labels, scores)),
            "ap": float(average_precision_score(labels, scores)),
            "auprc_trapezoid": float(auc(recall, precision)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def local_scores(reference, queries, neighbors=3, radius=1, top_fraction=0.05):
    """Match each block against nearby normal blocks; aggregate largest scores."""
    reference, queries = np.asarray(reference), np.asarray(queries)
    if reference.ndim != 4 or queries.ndim != 4 or reference.shape[1:] != queries.shape[1:]:
        raise ValueError("Local descriptors must have aligned N x H x W x D shapes")
    if not np.isfinite(reference).all() or not np.isfinite(queries).all():
        raise ValueError("Nonfinite local descriptors")
    if not 0 < top_fraction <= 1 or radius < 0 or neighbors < 1 or neighbors > len(reference):
        raise ValueError("Invalid patch-scoring configuration")
    _, height, width, _ = reference.shape
    maps = np.empty(queries.shape[:3], dtype=np.float64)
    for row in range(height):
        for column in range(width):
            bank = reference[:, max(0, row-radius):row+radius+1,
                             max(0, column-radius):column+radius+1, :].reshape(-1, reference.shape[-1])
            distances = cdist(queries[:, row, column, :], bank, metric="sqeuclidean")
            nearest = np.partition(distances, neighbors - 1, axis=1)[:, :neighbors]
            maps[:, row, column] = np.sqrt(np.maximum(nearest, 0)).mean(axis=1)
    top_count = math.ceil(height * width * top_fraction)
    flattened = maps.reshape(len(queries), -1)
    scores = np.partition(flattened, flattened.shape[1]-top_count, axis=1)[:, -top_count:].mean(axis=1)
    return finite_scores(scores), maps


def score_model(model, global_features, local_features):
    method = model["method"]
    if method == "global_hog_knn":
        scores = model["estimator"].kneighbors(global_features, return_distance=True)[0].mean(axis=1)
        return finite_scores(scores), None
    if method == "local_hog_knn":
        return local_scores(model["reference"], local_features, **model["settings"])
    if method == "hog_linear_svm":
        return finite_scores(model["estimator"].decision_function(global_features)), None
    raise ValueError(f"Unknown model {method}")


def fit_category(category, rows, global_features, local_features, protocol):
    normal_fit = select_indices(rows, category, "fit", 0)
    all_fit = select_indices(rows, category, "fit")
    validation = select_indices(rows, category, "validation")
    validation_labels = np.asarray([rows[i]["label"] for i in validation])
    models, validation_rows, search_rows = {}, [], []
    for method in METHODS:
        start = perf_counter()
        settings = protocol["models"][method]
        model = {"method": method, "category": category, "feature_protocol": protocol}
        if method == "global_hog_knn":
            estimator = NearestNeighbors(n_neighbors=settings["neighbors"], algorithm="brute", metric="euclidean", n_jobs=1)
            estimator.fit(global_features[normal_fit])
            model.update(estimator=estimator, fit_paths=[rows[i]["path"] for i in normal_fit])
        elif method == "local_hog_knn":
            model.update(reference=local_features[normal_fit].copy(),
                         settings={"neighbors": settings["neighbors"], "radius": settings["spatial_radius"], "top_fraction": settings["top_fraction"]},
                         fit_paths=[rows[i]["path"] for i in normal_fit])
        else:
            fit_labels = np.asarray([rows[i]["label"] for i in all_fit])
            best_ap = -np.inf
            for penalty in sorted(settings["C_candidates"]):
                estimator = make_pipeline(StandardScaler(), LinearSVC(C=penalty, class_weight="balanced",
                    random_state=protocol["seed"], dual="auto", tol=1e-4, max_iter=20000))
                estimator.fit(global_features[all_fit], fit_labels)
                if int(estimator[-1].n_iter_) >= estimator[-1].max_iter:
                    raise RuntimeError(f"SVM did not converge for {category}, C={penalty}")
                scores = finite_scores(estimator.decision_function(global_features[validation]))
                ap = float(average_precision_score(validation_labels, scores))
                search_rows.append({"category": category, "C": penalty, "validation_ap": ap})
                if ap > best_ap:
                    best_ap = ap
                    model.update(estimator=estimator, selected_C=penalty)
            model["fit_paths"] = [rows[i]["path"] for i in all_fit]
        scores, _ = score_model(model, global_features[validation], local_features[validation])
        threshold = (supervised_threshold(validation_labels, scores) if method == "hog_linear_svm" else
                     normal_threshold(scores[validation_labels == 0], protocol["normal_threshold_quantile"]))
        model["threshold"] = threshold
        model["validation_paths"] = [rows[i]["path"] for i in validation]
        model["fit_and_calibrate_seconds"] = perf_counter() - start
        model["validation_metrics"] = evaluate(validation_labels, scores, threshold)
        models[method] = model
        for index, score, prediction in zip(validation, scores, predict(scores, threshold)):
            validation_rows.append({"category": category, "method": method, "path": rows[index]["path"],
                "true_label": rows[index]["label"], "score": float(score), "threshold": threshold, "predicted_label": int(prediction)})
        print(f"Fitted {category}/{method}; threshold={threshold:.6g}", flush=True)
    return models, validation_rows, search_rows


def load_test_labels(data_root, rows, categories):
    path = data_root / "image_anomaly_labels.json"
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    expected = {row["path"] for row in rows if row["role"] == "test"}
    if set(data) != expected:
        raise ValueError(f"Test label keys mismatch: missing={sorted(expected-set(data))}, extra={sorted(set(data)-expected)}")
    output = {}
    for key, item in data.items():
        category = key.split("/")[0]
        if category not in categories or item.get("original_category") != category or item.get("label") not in {"good", "bad"}:
            raise ValueError(f"Invalid or misaligned test label: {key}")
        output[key] = int(item["label"] == "bad")
    return output, sha256(path)


def inventory_summary(rows, infos):
    counts = Counter((row["category"], row["role"], str(row["label"])) for row in rows)
    return {"images": len(rows), "counts_without_test_labels": {"/".join(k): v for k, v in sorted(counts.items())},
            "source_modes": dict(Counter(info["mode"] for info in infos)),
            "source_sizes": dict(Counter(f"{info['width']}x{info['height']}" for info in infos))}


def run(args):
    started = perf_counter()
    data_root = args.data_root.resolve()
    output = outside_data(args.output_dir, data_root)
    cache_dir = outside_data(args.cache_dir, data_root)
    if not data_root.is_dir():
        raise ValueError(f"Dataset root does not exist: {data_root}")
    output.mkdir(parents=True, exist_ok=False)
    events = []
    def event(name, **details):
        events.append({"event": name, "time": timestamp(), "elapsed_seconds": perf_counter()-started, **details})
        save_json(output / "events.json", events)
    try:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8-sig"))
        save_json(output / "protocol.json", protocol)
        event("protocol_registered", protocol_sha256=sha256(args.protocol))
        environment = {"python": sys.version, "platform": platform.platform(), "command": sys.argv,
            "packages": {name: version(name) for name in ["numpy", "scipy", "scikit-learn", "scikit-image", "Pillow", "matplotlib", "joblib", "threadpoolctl"]}}
        save_json(output / "environment.json", environment)
        code_hashes = {p.name: sha256(p) for p in ROOT.glob("*.py")}
        save_json(output / "provenance.json", {"created_at": timestamp(), "data_root": str(data_root),
            "protocol_sha256": sha256(args.protocol), "code_sha256": code_hashes, "output_dir": str(output)})
        rows = assign_roles(discover(data_root, protocol), protocol)
        global_features, local_features, infos, cache_info = load_features(data_root, rows, protocol, cache_dir, args.workers)
        save_csv(output / "inventory.csv", [{**row, **info} for row, info in zip(rows, infos)])
        save_csv(output / "splits.csv", rows)
        duplicates = duplicate_report(rows, infos)
        checks = {**inventory_summary(rows, infos), "duplicates": duplicates, "features": cache_info}
        save_json(output / "data_checks.json", checks)
        all_models, validation_rows, search_rows = {}, [], []
        (output / "models").mkdir()
        with threadpool_limits(limits=4):
            for category in protocol["categories"]:
                models, validation, searches = fit_category(category, rows, global_features, local_features, protocol)
                all_models[category] = models
                validation_rows.extend(validation)
                search_rows.extend(searches)
                for method, model in models.items():
                    joblib.dump(model, output / "models" / f"{category}_{method}.joblib", compress=3)
            save_csv(output / "validation_predictions.csv", validation_rows)
            save_csv(output / "svm_validation_search.csv", search_rows)
            frozen = {category: {method: {"threshold": model["threshold"],
                "selected_C": model.get("selected_C"), "fit_count": len(model["fit_paths"]),
                "validation_count": len(model["validation_paths"]), "validation_metrics": model["validation_metrics"],
                "fit_and_calibrate_seconds": model["fit_and_calibrate_seconds"],
                "model_sha256": sha256(output / "models" / f"{category}_{method}.joblib")}
                for method, model in models.items()} for category, models in all_models.items()}
            save_json(output / "frozen_models.json", frozen)
            event("all_models_and_thresholds_frozen", manifest_sha256=sha256(output / "frozen_models.json"))
            unlabelled, maps = [], {}
            for category, models in all_models.items():
                indices = select_indices(rows, category, "test")
                for method, model in models.items():
                    start = perf_counter()
                    scores, anomaly_maps = score_model(model, global_features[indices], local_features[indices])
                    frozen[category][method]["test_scoring_seconds"] = perf_counter() - start
                    if anomaly_maps is not None:
                        maps[category] = anomaly_maps
                    for index, score, prediction in zip(indices, scores, predict(scores, model["threshold"])):
                        unlabelled.append({"category": category, "method": method, "path": rows[index]["path"],
                            "score": float(score), "threshold": model["threshold"], "predicted_label": int(prediction)})
            save_csv(output / "unlabelled_test_predictions.csv", unlabelled)
            np.savez_compressed(output / "local_anomaly_maps.npz", **maps)
            event("unlabelled_test_predictions_saved")
            test_labels, label_hash = load_test_labels(data_root, rows, protocol["categories"])
            event("test_labels_loaded_for_evaluation", labels_sha256=label_hash)
            predictions = [{**row, "true_label": test_labels[row["path"]]} for row in unlabelled]
            save_csv(output / "predictions.csv", predictions)
            metrics = []
            for category in protocol["categories"]:
                for method in METHODS:
                    group = [r for r in predictions if r["category"] == category and r["method"] == method]
                    values = evaluate([r["true_label"] for r in group], [r["score"] for r in group], group[0]["threshold"])
                    metrics.append({"category": category, "method": method, "test_images": len(group), **values})
            save_csv(output / "metrics.csv", metrics)
            macro = {method: {name: float(np.mean([m[name] for m in metrics if m["method"] == method]))
                     for name in ["auroc", "ap", "auprc_trapezoid", "precision", "recall", "f1"]} for method in METHODS}
            unchanged = all(sha256(data_root / row["path"]) == info["sha256"] for row, info in zip(rows, infos))
            if not unchanged:
                raise RuntimeError("Original image hashes changed during the experiment")
            if sha256(data_root / "image_anomaly_labels.json") != label_hash:
                raise RuntimeError("Test label file changed during evaluation")
            summary = {"status": "complete", "created_at": timestamp(), "metrics": metrics,
                "macro_average": macro, "model_details": frozen, "data_checks": checks,
                "test_label_counts": {cat: dict(Counter(str(test_labels[r["path"]]) for r in rows if r["category"] == cat and r["role"] == "test")) for cat in protocol["categories"]},
                "original_images_unchanged": unchanged, "phase_order_verified": True,
                "elapsed_before_plots_seconds": perf_counter()-started,
                "main_method_prespecified": "local_hog_knn"}
            save_json(output / "summary.json", summary)
        from plot_results import make_plots
        make_plots(output, data_root, predictions, maps, protocol)
        event("evaluation_and_plots_complete")
        summary["total_seconds"] = perf_counter()-started
        save_json(output / "summary.json", summary)
        print(json.dumps({"output": str(output), "metrics": metrics, "macro_average": macro}, ensure_ascii=False, indent=2), flush=True)
        return output
    except Exception as error:
        import traceback
        save_json(output / "failure.json", {"error": str(error), "traceback": traceback.format_exc(), "time": timestamp()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent.parent / "DM_2026_Dataset" / "Image_Anomaly_Detection" / "Image_Anomaly_Detection")
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiment_protocol.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    run(args)


if __name__ == "__main__":
    main()
