"""Compare original T3 model vs a CatBoost variant.

This script intentionally reuses the *same* data pipeline as `main.py`:
- training set: `DatabaseManager.get_train_databases()`
- evaluation slices: same as `src/figures/accuracy_table.py`

Run from repo root:

  . venv/bin/activate
  python -m src.compare_catboost

To evaluate the compiled CatBoost variant:

  T3_RUN_CAT_COMPILED=1 python -m src.compare_catboost

If CatBoost isn't installed:

  pip install catboost
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.evaluation import QueryEstimationCache
from src.features import FeatureMapper
from src.metrics import q_error
from src.model import Model
from src.optimizer import QueryCategory
from src.train import optimize_all


try:
    from catboost import CatBoostRegressor  # type: ignore

    _HAS_CATBOOST = True
except Exception:
    CatBoostRegressor = None  # type: ignore
    _HAS_CATBOOST = False


@dataclass(frozen=True)
class SummaryRow:
    dataset: str
    p50: float
    p90: float
    avg: float


def _build_accuracy_slices(predicted_cardinalities: bool = False):
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


def _summarize_qerrors(estimation_cache: QueryEstimationCache, queries) -> SummaryRow:
    estimates = [estimation_cache.queries[q.name].estimated_time for q in queries]
    q_errors = np.array([q_error(q.get_total_runtime(), est) for q, est in zip(queries, estimates)], dtype=float)
    return SummaryRow(dataset="", p50=float(np.median(q_errors)), p90=float(np.quantile(q_errors, 0.9)), avg=float(q_errors.mean()))


def _evaluate_model(model: Model, predicted_cardinalities: bool = False) -> tuple[pd.DataFrame, float]:
    """Returns (accuracy_table_df, avg_inference_ms_per_query)."""
    cache = QueryEstimationCache(model, predicted_cardinalities)

    rows = []
    for dataset_name, queries in _build_accuracy_slices(predicted_cardinalities):
        summary = _summarize_qerrors(cache, queries)
        rows.append({"Dataset": dataset_name, "p50": summary.p50, "p90": summary.p90, "Avg": summary.avg})
    df = pd.DataFrame(rows)

    all_queries = DataCollector.collect_benchmarks(DatabaseManager.get_all_databases(), predicted_cardinalities)
    start = time.perf_counter()
    for q in all_queries:
        _ = model.estimate_runtime(q)
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / max(1, len(all_queries))) * 1000.0
    return df, avg_ms


def _build_per_tuple_training_data(benchmarks, feature_mapper: FeatureMapper) -> tuple[np.ndarray, np.ndarray]:
    x_vectors: list[np.ndarray] = []
    y_values: list[float] = []
    for query in benchmarks:
        for x, y in query.get_per_tuple_pipeline_runtime_data(feature_mapper):
            if np.any(x != 0):
                x_vectors.append(x)
                y_values.append(float(y))
    x = np.vstack(x_vectors).astype(np.float32, copy=False)
    y = np.array(y_values, dtype=np.float32)
    y = np.maximum(y, 1e-15)
    y = -np.log(y)
    y = np.maximum(y, 1e-6)
    return x, y


def _benchmark_batch_predict(predict_fn, x: np.ndarray, repeats: int = 5) -> float:
    """Return best-case avg microseconds per row for predict_fn(x)."""
    best_s = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        _ = predict_fn(x)
        elapsed = time.perf_counter() - start
        best_s = elapsed if best_s is None else min(best_s, elapsed)
    us_per_row = (best_s / max(1, x.shape[0])) * 1e6
    return float(us_per_row)

def _benchmark_batch_predict_fixed(predict_fn, *, n_rows: int, repeats: int = 5) -> float:
    best_s = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        _ = predict_fn()
        elapsed = time.perf_counter() - start
        best_s = elapsed if best_s is None else min(best_s, elapsed)
    us_per_row = (best_s / max(1, n_rows)) * 1e6
    return float(us_per_row)

class CatBoostPerTupleModel(Model):
    """Drop-in `Model` that mirrors `PerTupleTreeModel` but uses CatBoost for the per-tuple predictor."""

    def __init__(self, regressor):
        self.regressor = regressor
        self._feature_mapper = FeatureMapper()

    def get_feature_mapper(self) -> FeatureMapper:
        return self._feature_mapper

    def _predict_pipeline_times(self, x: np.ndarray, scan_sizes: np.ndarray) -> np.ndarray:
        mask = np.any(x != 0, axis=1)
        pred_log = self.regressor.predict(x).flatten()
        pred = np.exp(-pred_log)
        scan_sizes = np.array(scan_sizes, copy=True)
        scan_sizes[scan_sizes < 1] = 1
        pred = pred * scan_sizes
        pred *= mask
        pred[pred < 0] = 0.0
        return pred

    def estimate_runtime(self, query) -> float:
        return sum(self.estimate_pipeline_runtime(query))

    def estimate_pipeline_runtime(self, query) -> list[float]:
        x = query.get_feature_matrix(self._feature_mapper)
        scan_sizes = self._feature_mapper.get_pipeline_scan_sizes(query.query_plan)
        pred = self._predict_pipeline_times(np.array(x, dtype=np.float32), scan_sizes)
        return [max(0.0, float(e)) for e in pred]


class CompiledCatBoostPerTupleModel(Model):
    """Drop-in `Model` that uses a natively compiled C++ CatBoost model."""

    def __init__(self, lib_path: Path):
        self._feature_mapper = FeatureMapper()
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.predict_batch.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            ctypes.c_int
        ]
        self.lib.predict_batch.restype = None

    def get_feature_mapper(self) -> FeatureMapper:
        return self._feature_mapper

    def _predict_pipeline_times(self, x: np.ndarray, scan_sizes: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        n_rows, n_cols = x.shape
        out = np.zeros(n_rows, dtype=np.float32)
        
        self.lib.predict_batch(x, out, n_rows, n_cols)
        
        mask = np.any(x != 0, axis=1)
        pred_log = out
        pred = np.exp(-pred_log)
        scan_sizes = np.array(scan_sizes, copy=True)
        scan_sizes[scan_sizes < 1] = 1
        pred = pred * scan_sizes
        pred *= mask
        pred[pred < 0] = 0.0
        return pred

    def estimate_runtime(self, query) -> float:
        return sum(self.estimate_pipeline_runtime(query))

    def estimate_pipeline_runtime(self, query) -> list[float]:
        x = query.get_feature_matrix(self._feature_mapper)
        scan_sizes = self._feature_mapper.get_pipeline_scan_sizes(query.query_plan)
        pred = self._predict_pipeline_times(np.array(x, dtype=np.float32), scan_sizes)
        return [max(0.0, float(e)) for e in pred]


def train_catboost_per_tuple_model(predicted_cardinalities: bool = False) -> CatBoostPerTupleModel:
    if not _HAS_CATBOOST:
        raise RuntimeError(
            "CatBoost is not installed. Install it with: pip install catboost\n"
            "(then rerun: python -m src.compare_catboost)"
        )

    feature_mapper = FeatureMapper()
    train_benchmarks = DataCollector.collect_benchmarks(DatabaseManager.get_train_databases(), predicted_cardinalities)
    x, y = _build_per_tuple_training_data(train_benchmarks, feature_mapper)

    seed = 21
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)

    reg = CatBoostRegressor(
        iterations=200,
        learning_rate=0.1,
        loss_function="RMSE",
        eval_metric="MAPE",
        grow_policy="SymmetricTree",
        depth=5,
        random_seed=seed,
        thread_count=-1,
        verbose=False,
    )
    reg.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=False)
    return CatBoostPerTupleModel(reg)

def compile_catboost_to_cpp(cat_model: CatBoostPerTupleModel, cache_dir: Path) -> CompiledCatBoostPerTupleModel:
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cpp_model_path = cache_dir / "catboost_model.cpp"
    cat_model.regressor.save_model(str(cpp_model_path), format="CPP")
    
    wrapper_path = cache_dir / "catboost_wrapper.cpp"
    wrapper_code = """
#include <vector>

double ApplyCatboostModel(const std::vector<float>& floatFeatures);

extern "C" {
    void predict_batch(const float* x, float* out, int n_rows, int n_cols) {
        std::vector<float> features(n_cols);
        for (int i = 0; i < n_rows; ++i) {
            for (int j = 0; j < n_cols; ++j) {
                features[j] = x[i * n_cols + j];
            }
            out[i] = (float)ApplyCatboostModel(features);
        }
    }
}
"""
    wrapper_path.write_text(wrapper_code)
    
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    lib_path = cache_dir / f"libcatboost_compiled{ext}"
    
    if not lib_path.exists() or os.environ.get("T3_FORCE_RECOMPILE", "0") == "1":
        compiler = os.environ.get("CXX", "clang++" if sys.platform == "darwin" else "g++")
        cmd = [
            compiler, "-O3", "-shared", "-fPIC", "-std=c++11",
            str(wrapper_path), str(cpp_model_path),
            "-o", str(lib_path)
        ]
        subprocess.check_call(cmd)
    
    return CompiledCatBoostPerTupleModel(lib_path)

def main():
    predicted_cardinalities = False
    run_cat_compiled = os.environ.get("T3_RUN_CAT_COMPILED", "0") == "1"

    print("--- Comparing: Original T3 vs CatBoost (same data pipeline) ---")

    start = time.perf_counter()
    t3_model = optimize_all(predicted_cardinalities)
    t3_train_s = time.perf_counter() - start

    # Build a stable matrix for predictor-only benchmarks
    feature_mapper = FeatureMapper()
    train_benchmarks = DataCollector.collect_benchmarks(DatabaseManager.get_train_databases(), predicted_cardinalities)
    bench_x, _bench_y = _build_per_tuple_training_data(train_benchmarks, feature_mapper)

    t3_table, t3_inf_ms = _evaluate_model(t3_model, predicted_cardinalities)

    print("\nT3 (original) — Q-Error table")
    print(t3_table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"[Timing] Train: {t3_train_s:.2f}s | Avg inference: {t3_inf_ms:.3f} ms/query")
    t3_us_row = _benchmark_batch_predict(lambda m: t3_model.tree.predict(m), bench_x)
    print(f"[Model-only] LightGBM predict: {t3_us_row:.3f} us/row")

    if _HAS_CATBOOST:
        start = time.perf_counter()
        cat_model = train_catboost_per_tuple_model(predicted_cardinalities)
        cat_train_s = time.perf_counter() - start

        cat_table, cat_inf_ms = _evaluate_model(cat_model, predicted_cardinalities)
        print("\nCatBoost (per-tuple) — Q-Error table")
        print(cat_table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print(f"[Timing] Train: {cat_train_s:.2f}s | Avg inference: {cat_inf_ms:.3f} ms/query")

        cat_us_row = _benchmark_batch_predict(lambda m: cat_model.regressor.predict(m), bench_x)
        print(f"[Model-only] CatBoost predict: {cat_us_row:.3f} us/row")
        
        if run_cat_compiled:
            start = time.perf_counter()
            cat_compiled = compile_catboost_to_cpp(cat_model, Path("compare_output") / "compiled")
            cat_compile_s = time.perf_counter() - start
            
            catc_table, catc_inf_ms = _evaluate_model(cat_compiled, predicted_cardinalities)
            print("\nCatBoost (compiled C++) — Q-Error table")
            print(catc_table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
            print(f"[Timing] Compile: {cat_compile_s:.2f}s | Avg inference: {catc_inf_ms:.3f} ms/query")
            
            bench_x_contig = np.ascontiguousarray(bench_x, dtype=np.float32)
            n_rows, n_cols = bench_x_contig.shape
            out = np.zeros(n_rows, dtype=np.float32)
            
            catc_us_row = _benchmark_batch_predict_fixed(
                lambda: cat_compiled.lib.predict_batch(bench_x_contig, out, n_rows, n_cols),
                n_rows=n_rows
            )
            speedup = cat_us_row / max(1e-12, catc_us_row)
            print(f"[Model-only] Compiled CatBoost predict: {catc_us_row:.3f} us/row (x{speedup:.2f} speedup)")

    else:
        print("\n[WARN] CatBoost not installed — skipping CatBoost comparison.")
        print("       Install with: pip install catboost")


if __name__ == "__main__":
    main()
