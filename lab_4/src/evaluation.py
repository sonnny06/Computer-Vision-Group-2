"""TV6 binary evaluation: label 1 means similar; smaller distance is better."""

import numbers

import numpy as np


def _numeric_vector(values, name):
    # Preserve input types so mixed strings/numbers cannot be silently coerced.
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty 1D sequence")
    if any(isinstance(value, (bool, np.bool_)) or
           not isinstance(value, numbers.Real) for value in array):
        raise ValueError(f"{name} must contain real numbers, not bools")
    return array


def _labels(values, name):
    array = _numeric_vector(values, name)
    if any(value not in (0, 1) for value in array):
        raise ValueError(f"{name} must contain only numeric labels 0 or 1")
    return array.astype(int)


def _bit_counts(values, name):
    array = _numeric_vector(values, name)
    if any(not np.isfinite(value) or value < 0 or value != int(value)
           for value in array):
        raise ValueError(f"{name} must contain finite nonnegative integer bit counts")
    return array


def _matching_lengths(first, second):
    if len(first) != len(second):
        raise ValueError("Inputs must have the same length")


def classification_metrics(y_true, y_pred):
    """Return confusion counts and four ratios; undefined ratios are NaN."""
    truth = _labels(y_true, "y_true")
    prediction = _labels(y_pred, "y_pred")
    _matching_lengths(truth, prediction)
    tp = int(np.count_nonzero((truth == 1) & (prediction == 1)))
    fp = int(np.count_nonzero((truth == 0) & (prediction == 1)))
    tn = int(np.count_nonzero((truth == 0) & (prediction == 0)))
    fn = int(np.count_nonzero((truth == 1) & (prediction == 0)))

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else float("nan")

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "accuracy": (tp + tn) / len(truth),
            "recall": ratio(tp, tp + fn),
            "specificity": ratio(tn, tn + fp),
            "precision": ratio(tp, tp + fp)}


def evaluate_thresholds(y_true, distances, thresholds):
    """Return one metrics row per threshold, using distance <= threshold."""
    truth = _labels(y_true, "y_true")
    distances = _bit_counts(distances, "distances")
    thresholds = _bit_counts(thresholds, "thresholds")
    _matching_lengths(truth, distances)
    return [{"threshold": int(threshold), "comparison": "<=",
             **classification_metrics(truth, (distances <= threshold).astype(int))}
            for threshold in thresholds]


def roc_auc_from_distances(y_true, distances):
    """Exact ROC with score=-distance, grouped ties and trapezoidal AUC.

    The initial distance threshold is -inf (no predicted positives). Both
    classes are required; a one-class dataset raises ValueError.
    """
    truth = _labels(y_true, "y_true")
    distances = _bit_counts(distances, "distances")
    _matching_lengths(truth, distances)
    positives = int(np.count_nonzero(truth == 1))
    negatives = len(truth) - positives
    if not positives or not negatives:
        raise ValueError("ROC/AUC is undefined: y_true must contain both classes")
    thresholds = sorted(set(int(value) for value in distances))
    fpr, tpr = [0.0], [0.0]
    for threshold in thresholds:
        selected = distances <= threshold
        tpr.append(int(np.count_nonzero(selected & (truth == 1))) / positives)
        fpr.append(int(np.count_nonzero(selected & (truth == 0))) / negatives)
    auc = sum((fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
              for i in range(1, len(fpr)))
    return {"fpr": fpr, "tpr": tpr,
            "thresholds": [float("-inf"), *thresholds], "auc": auc}
