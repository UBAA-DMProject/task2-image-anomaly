"""Plots and deterministic case selection from frozen predictions."""

from pathlib import Path
import math
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

NAMES = {"global_hog_knn": "Global HOG kNN", "local_hog_knn": "Local HOG kNN", "hog_linear_svm": "HOG linear SVM"}
COLORS = {"global_hog_knn": "#526D82", "local_hog_knn": "#D97732", "hog_linear_svm": "#35826B"}


def finish(figure, path):
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def select_cases(group):
    """Pick the highest-scoring FP, lowest-scoring FN, then fixed-path TP/TN."""
    pools = {
        "FP": sorted([r for r in group if r["true_label"] == 0 and r["predicted_label"] == 1], key=lambda r: (-r["score"], r["path"])),
        "FN": sorted([r for r in group if r["true_label"] == 1 and r["predicted_label"] == 0], key=lambda r: (r["score"], r["path"])),
        "TP": sorted([r for r in group if r["true_label"] == 1 and r["predicted_label"] == 1], key=lambda r: r["path"]),
        "TN": sorted([r for r in group if r["true_label"] == 0 and r["predicted_label"] == 0], key=lambda r: r["path"]),
    }
    return [(kind, row) for kind, pool in pools.items() for row in pool[:2]]


def make_plots(output, data_root, predictions, maps, protocol):
    from run_experiment import save_csv, evaluate
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    cases = []
    for category in protocol["categories"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for method, name in NAMES.items():
            group = [r for r in predictions if r["category"] == category and r["method"] == method]
            labels = np.array([r["true_label"] for r in group])
            scores = np.array([r["score"] for r in group])
            metrics = evaluate(labels, scores, group[0]["threshold"])
            fpr, tpr, _ = roc_curve(labels, scores)
            precision, recall, _ = precision_recall_curve(labels, scores)
            axes[0].plot(fpr, tpr, color=COLORS[method], label=f"{name}: {metrics['auroc']:.3f}")
            axes[1].step(recall, precision, where="post", color=COLORS[method], label=f"{name}: {metrics['ap']:.3f}")
        axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        axes[1].axhline(labels.mean(), linestyle="--", color="gray", linewidth=1)
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title=f"{category}: ROC (AUROC)", xlim=(0, 1), ylim=(0, 1.02))
        axes[1].set(xlabel="Recall", ylabel="Precision", title=f"{category}: PR (AP)", xlim=(0, 1), ylim=(0, 1.02))
        for axis in axes:
            axis.legend(fontsize=8, loc="lower right" if axis is axes[0] else "lower left")
        fig.tight_layout()
        finish(fig, figures / f"{category}_curves.png")
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        for axis, (method, name) in zip(axes, NAMES.items()):
            group = [r for r in predictions if r["category"] == category and r["method"] == method]
            ConfusionMatrixDisplay.from_predictions([r["true_label"] for r in group], [r["predicted_label"] for r in group],
                labels=[0, 1], display_labels=["Normal", "Anomaly"], ax=axis, colorbar=False, cmap="Blues")
            axis.set_title(name, fontsize=10)
        fig.suptitle(category)
        fig.tight_layout()
        finish(fig, figures / f"{category}_confusion.png")
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        for axis, (method, name) in zip(axes, NAMES.items()):
            group = [r for r in predictions if r["category"] == category and r["method"] == method]
            scores = np.array([r["score"] for r in group])
            bins = np.histogram_bin_edges(scores, bins=15)
            for label, display in [(0, "Normal"), (1, "Anomaly")]:
                axis.hist([r["score"] for r in group if r["true_label"] == label], bins=bins, label=display, alpha=0.6)
            axis.axvline(group[0]["threshold"], color="black", linestyle="--", label="Validation threshold")
            axis.set(title=name, xlabel="Anomaly score", ylabel="Images")
            axis.legend(fontsize=7)
        fig.suptitle(category)
        fig.tight_layout()
        finish(fig, figures / f"{category}_scores.png")
        group = [r for r in predictions if r["category"] == category and r["method"] == "local_hog_knn"]
        selected = select_cases(group)
        test_order = {row["path"]: i for i, row in enumerate(group)}
        fig, axes = plt.subplots(len(selected), 2, figsize=(6.3, 2.7 * len(selected)), squeeze=False)
        for axes_row, (kind, row) in zip(axes, selected):
            with Image.open(data_root / row["path"]) as source:
                image = source.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
            anomaly_map = maps[category][test_order[row["path"]]]
            axes_row[0].imshow(image)
            axes_row[0].set_title(f"{kind} | {Path(row['path']).name} | score={row['score']:.4f}", fontsize=9)
            axes_row[1].imshow(image)
            axes_row[1].imshow(anomaly_map, cmap="inferno", interpolation="bilinear", alpha=0.55, extent=(8, 248, 248, 8))
            axes_row[1].set_xlim(-0.5, 255.5)
            axes_row[1].set_ylim(255.5, -0.5)
            axes_row[1].set_title("Relative block scores (per-image scale)", fontsize=9)
            for axis in axes_row:
                axis.axis("off")
            cases.append({"category": category, "case_type": kind, **row})
        fig.suptitle(f"{category}: prespecified local detector; qualitative maps", y=1.002, fontsize=11)
        fig.tight_layout()
        finish(fig, figures / f"{category}_cases.png")
    save_csv(output / "selected_cases.csv", cases)

