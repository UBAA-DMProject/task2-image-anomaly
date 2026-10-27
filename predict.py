"""Score new images with a saved model without reading labels or refitting."""

import argparse
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from run_experiment import extract_image, outside_data, predict, save_csv, score_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    source = args.input_path.resolve()
    paths = ([source] if source.is_file() else sorted(p for p in source.rglob("*")
             if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}))
    if not paths:
        parser.error("No input images found")
    output = outside_data(args.output_csv, source.parent if source.is_file() else source)
    if output.exists():
        parser.error("Output CSV already exists")
    model = joblib.load(args.model)
    extracted = [extract_image(path, model["feature_protocol"]) for path in paths]
    with threadpool_limits(limits=4):
        scores, _ = score_model(model, np.stack([x[0] for x in extracted]), np.stack([x[1] for x in extracted]))
    predictions = predict(scores, model["threshold"])
    save_csv(output, [{"path": str(path), "category": model["category"], "method": model["method"],
                      "score": float(score), "threshold": model["threshold"], "predicted_label": int(label)}
                     for path, score, label in zip(paths, scores, predictions)])
    print(f"Scored {len(paths)} images with {model['category']}/{model['method']}: {output}")


if __name__ == "__main__":
    main()
