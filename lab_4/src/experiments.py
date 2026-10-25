"""Small TV6 experiments reusing the TV4 hash and Hamming helpers."""

from pathlib import Path

import numpy as np

from lab_4.src.evaluation import (
    _bit_counts, _labels, evaluate_thresholds, roc_auc_from_distances,
)
from lab_4.src.wavelet_hash import hamming_distance, wavelet_hash


def run_experiments(pairs, image_root, configs, thresholds):
    """Evaluate configurations on the same unique labelled image pairs.

    ``pairs`` accepts CSV DictReader rows. Only this adapter converts literal
    CSV labels '0'/'1' to numbers. Image paths are relative to ``image_root``
    (normally lab_4). Invalid rows/images stop the run rather than being skipped.
    """
    thresholds = _bit_counts(thresholds, "thresholds").tolist()
    rows, seen = [], set()
    image_root = Path(image_root)
    for index, pair in enumerate(pairs, start=1):
        try:
            image1, image2 = pair["image1"], pair["image2"]
            if not all(isinstance(path, str) and path.strip()
                       for path in (image1, image2)):
                raise ValueError("image1 and image2 must be nonempty path strings")
            label = pair["label"]
            if isinstance(label, str) and label in ("0", "1"):
                label = int(label)
            label = int(_labels([label], "label")[0])
            key = tuple(sorted((str((image_root / image1).resolve()),
                                str((image_root / image2).resolve()))))
            if key in seen:
                raise ValueError("duplicate image pair (including reversed order)")
            seen.add(key)
            rows.append({"image1": image1, "image2": image2, "label": label,
                         "notes": pair.get("notes", "")})
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Pair row {index}: {error}") from error
    if not rows:
        raise ValueError("pairs must not be empty")
    configs = list(configs)
    if not configs:
        raise ValueError("configs must not be empty")
    config_ids = set()
    for config in configs:
        config_id = config.get("config_id")
        if not isinstance(config_id, str) or not config_id.strip() or config_id in config_ids:
            raise ValueError("Every configuration needs a unique nonempty config_id")
        config_ids.add(config_id)
        required = {"config_id", "wavelet", "level", "mode", "hash_size", "image_scale"}
        if set(config) != required:
            raise ValueError(f"Configuration {config_id}: expected keys {sorted(required)}")
        for name in ("hash_size", "level", "image_scale"):
            value = config[name]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"Configuration {config_id}: {name} must be a positive integer")

    result = {"pair_scores": [], "metrics_by_threshold": [],
              "configuration_summary": [], "roc_curves": {}}
    truth = [row["label"] for row in rows]
    n_positive = sum(truth)
    image_names = dict.fromkeys(name for row in rows
                               for name in (row["image1"], row["image2"]))
    for config in configs:
        config_id = config["config_id"]
        parameters = {key: value for key, value in config.items() if key != "config_id"}
        hashes = {}
        expected_bits = config["hash_size"] ** 2
        for name in image_names:
            try:
                hashes[name] = wavelet_hash(image_root / name, **parameters)
                if hashes[name].bit_length != expected_bits:
                    raise ValueError(f"expected {expected_bits} hash bits, got {hashes[name].bit_length}")
            except Exception as error:
                raise ValueError(f"Configuration {config_id}, image {name}: {error}") from error
        distances = []
        for row in rows:
            try:
                distance = hamming_distance(hashes[row["image1"]], hashes[row["image2"]])
                distance = int(_bit_counts([distance], "distance")[0])
                if distance > expected_bits:
                    raise ValueError(f"distance exceeds hash length {expected_bits}")
            except Exception as error:
                raise ValueError(f"Configuration {config_id}, pair {row['image1']} / {row['image2']}: {error}") from error
            distances.append(distance)
            result["pair_scores"].append({"config_id": config_id, **row,
                                          "distance": distance, "bit_length": expected_bits,
                                          "score": -distance})
        result["metrics_by_threshold"].extend(
            {"config_id": config_id, **metrics}
            for metrics in evaluate_thresholds(truth, distances, thresholds))
        auc = float("nan")
        if 0 < n_positive < len(rows):
            roc = roc_auc_from_distances(truth, distances)
            result["roc_curves"][config_id] = roc
            auc = roc["auc"]
        result["configuration_summary"].append(
            {**config, "n_pairs": len(rows), "n_positive": n_positive,
             "n_negative": len(rows) - n_positive, "auc": auc})
    return result
