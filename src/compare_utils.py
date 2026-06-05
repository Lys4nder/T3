from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd

from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.evaluation import QueryEstimationCache
from src.metrics import q_error
from src.optimizer import QueryCategory


def build_accuracy_slices(predicted_cardinalities: bool = False):
    return [
        (
            "Train Queries",
            DataCollector.collect_benchmarks(DatabaseManager.get_train_databases(), predicted_cardinalities),
        ),
        (
            "All TPC-DS Test Queries",
            DataCollector.collect_benchmarks(DatabaseManager.get_test_databases(), predicted_cardinalities),
        ),
        (
            "TPC-DS Benchmark Queries",
            DataCollector.collect_benchmarks(
                DatabaseManager.get_test_databases(),
                predicted_cardinalities,
                query_category=[QueryCategory.fixed],
            ),
        ),
        (
            "TPC-DS sf 100 Test Queries",
            DataCollector.collect_benchmarks([DatabaseManager.get_database("tpcdsSf100")], predicted_cardinalities),
        ),
        (
            "TPC-DS sf 100 Benchmark Queries",
            DataCollector.collect_benchmarks(
                [DatabaseManager.get_database("tpcdsSf100")],
                predicted_cardinalities,
                query_category=[QueryCategory.fixed],
            ),
        ),
    ]


def evaluate_model(model, predicted_cardinalities: bool = False) -> tuple[pd.DataFrame, float]:
    """Returns (accuracy_table_df, avg_inference_ms_per_query)."""

    cache = QueryEstimationCache(model, predicted_cardinalities)

    rows = []
    for dataset_name, queries in build_accuracy_slices(predicted_cardinalities):
        estimates = [cache.queries[q.name].estimated_time for q in queries]
        q_errors = np.array([q_error(q.get_total_runtime(), est) for q, est in zip(queries, estimates)], dtype=float)
        rows.append(
            {
                "Dataset": dataset_name,
                "p50": float(np.median(q_errors)),
                "p90": float(np.quantile(q_errors, 0.9)),
                "Avg": float(q_errors.mean()),
                "Max": float(q_errors.max()),
            }
        )

    df = pd.DataFrame(rows)

    all_queries = DataCollector.collect_benchmarks(DatabaseManager.get_all_databases(), predicted_cardinalities)
    start = time.perf_counter()
    for q in all_queries:
        _ = model.estimate_runtime(q)
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / max(1, len(all_queries))) * 1000.0

    return df, float(avg_ms)


def benchmark_batch_predict(predict_fn, x: np.ndarray, repeats: int = 5) -> float:
    """Return best-case avg microseconds per row for predict_fn(x)."""

    best_s = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        _ = predict_fn(x)
        elapsed = time.perf_counter() - start
        best_s = elapsed if best_s is None else min(best_s, elapsed)

    us_per_row = (best_s / max(1, x.shape[0])) * 1e6
    return float(us_per_row)


def benchmark_batch_predict_fixed(predict_fn, *, n_rows: int, repeats: int = 5) -> float:
    """Return best-case avg microseconds per row for a fixed-argument predict_fn()."""

    best_s = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        _ = predict_fn()
        elapsed = time.perf_counter() - start
        best_s = elapsed if best_s is None else min(best_s, elapsed)

    us_per_row = (best_s / max(1, n_rows)) * 1e6
    return float(us_per_row)
