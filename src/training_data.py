from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def build_per_tuple_training_data(
    benchmarks: Iterable[object],
    feature_mapper: object,
    *,
    dtype: type = np.float32,
    min_y: float = 1e-15,
    min_log_y: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for per-tuple runtime regression.

    Expects each benchmark query to implement:
      - get_per_tuple_pipeline_runtime_data(feature_mapper) -> iterable[(x_vec, y_val)]

    Mirrors the original T3 training transformation:
      - filter out all-zero feature rows
      - clamp y to avoid log(0)
      - y := -log(y)
      - clamp transformed targets to avoid degenerate values
    """

    x_vectors: list[np.ndarray] = []
    y_values: list[float] = []

    for query in benchmarks:
        for x, y in query.get_per_tuple_pipeline_runtime_data(feature_mapper):
            if np.any(x != 0):
                x_vectors.append(x)
                y_values.append(float(y))

    if len(x_vectors) == 0:
        raise ValueError("No per-tuple training rows found (all feature rows were zero?)")

    x = np.vstack(x_vectors).astype(dtype, copy=False)
    y = np.array(y_values, dtype=dtype)

    y = np.maximum(y, min_y)
    y = -np.log(y)
    y = np.maximum(y, min_log_y)

    return x, y
