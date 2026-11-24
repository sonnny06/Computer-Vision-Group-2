
import time

import numpy as np
import pandas as pd


def measure_runtime(function, *args, repeats=3, **kwargs):
    if not callable(function):
        raise TypeError("function must be callable")

    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")

    times = []
    output = None

    for _ in range(repeats):
        start = time.perf_counter()
        output = function(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return {
        "output": output,
        "average_time_ms": sum(times) / len(times),
        "min_time_ms": min(times),
        "max_time_ms": max(times),
    }


def compare_methods(image, methods, repeats=3):
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.size == 0:
        raise ValueError("image cannot be empty")

    if not isinstance(methods, dict) or not methods:
        raise ValueError("methods must be a non-empty dictionary")

    outputs = {}
    results = []

    for name, function in methods.items():
        measurement = measure_runtime(
            function,
            image.copy(),
            repeats=repeats,
        )

        outputs[name] = measurement["output"]

        results.append({
            "method": name,
            "average_time_ms": measurement["average_time_ms"],
            "min_time_ms": measurement["min_time_ms"],
            "max_time_ms": measurement["max_time_ms"],
        })

    return outputs, pd.DataFrame(results)


def save_results(results, output_path):
    if isinstance(results, list):
        results = pd.DataFrame(results)

    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a list or pandas DataFrame")

    results.to_csv(output_path, index=False)
