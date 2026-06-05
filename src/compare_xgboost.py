from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.features import FeatureMapper
from src.model import Model
from src.train import optimize_all

from src.compare_utils import (
    benchmark_batch_predict as _benchmark_batch_predict,
    benchmark_batch_predict_fixed as _benchmark_batch_predict_fixed,
    evaluate_model as _evaluate_model,
)
from src.training_data import build_per_tuple_training_data as _build_per_tuple_training_data


try:
    import lleaves  # type: ignore

    _HAS_LLEAVES = True
except Exception:
    lleaves = None  # type: ignore
    _HAS_LLEAVES = False


try:
    import treelite  # type: ignore
    import tl2cgen  # type: ignore

    _HAS_TL2CGEN = True
except Exception:
    treelite = None  # type: ignore
    tl2cgen = None  # type: ignore
    _HAS_TL2CGEN = False


try:
    from xgboost import XGBRegressor  # type: ignore

    _HAS_XGBOOST = True
except Exception:
    XGBRegressor = None  # type: ignore
    _HAS_XGBOOST = False


class XGBPerTupleModel(Model):
    """Drop-in `Model` that mirrors `PerTupleTreeModel` but uses XGBoost for the per-tuple predictor."""

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


class LleavesPerTupleModel(Model):
    """Drop-in `Model` like `PerTupleTreeModel`, but uses compiled lleaves for prediction."""

    def __init__(self, compiled_model):
        self.compiled_model = compiled_model
        self._feature_mapper = FeatureMapper()

    def get_feature_mapper(self) -> FeatureMapper:
        return self._feature_mapper

    def _predict_pipeline_times(self, x: np.ndarray, scan_sizes: np.ndarray) -> np.ndarray:
        mask = np.any(x != 0, axis=1)
        pred_log = np.asarray(self.compiled_model.predict(x), dtype=np.float64).reshape(-1)
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


class TL2CGenXGBPerTupleModel(Model):
    """Drop-in `Model` like `PerTupleTreeModel`, but uses a compiled tl2cgen predictor."""

    def __init__(self, predictor):
        self.predictor = predictor
        self._feature_mapper = FeatureMapper()

    def get_feature_mapper(self) -> FeatureMapper:
        return self._feature_mapper

    def _predict_pipeline_times(self, x: np.ndarray, scan_sizes: np.ndarray) -> np.ndarray:
        mask = np.any(x != 0, axis=1)
        dmat = tl2cgen.DMatrix(x)
        pred_log = np.asarray(self.predictor.predict(dmat), dtype=np.float64).reshape(-1)
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


def compile_t3_with_lleaves(t3_model: Model, cache_dir: Path) -> LleavesPerTupleModel:
    if not _HAS_LLEAVES:
        raise RuntimeError("lleaves is not installed. Install it with: pip install lleaves")

    cache_dir.mkdir(parents=True, exist_ok=True)

    model_file = cache_dir / "t3_lightgbm_model.txt"
    # Save LightGBM booster in a format lleaves can consume
    t3_model.tree.save_model(str(model_file))

    cache_file = cache_dir / "t3_lleaves_cache.bin"
    compiled = lleaves.Model(str(model_file))
    compiled.compile(cache=str(cache_file))
    return LleavesPerTupleModel(compiled)


def compile_xgboost_with_tl2cgen(regressor, cache_dir: Path) -> TL2CGenXGBPerTupleModel:
    """Compile an XGBoost regressor into a shared library and load it as a tl2cgen Predictor."""

    if not _HAS_TL2CGEN:
        raise RuntimeError(
            "tl2cgen/treelite not available. Install with: pip install tl2cgen\n"
            "On macOS you may also need OpenMP runtime (libomp) and DYLD_LIBRARY_PATH set."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)

    ext = ".dylib" if sys.platform == "darwin" else ".so"
    libpath = cache_dir / f"xgboost_tl2cgen{ext}"
    toolchain = os.environ.get(
        "T3_TL2CGEN_TOOLCHAIN",
        "clang" if sys.platform == "darwin" else "gcc",
    )

    # Recompile only if missing (or if forced)
    if (not libpath.exists()) or os.environ.get("T3_TL2CGEN_FORCE_RECOMPILE", "0") == "1":
        tl_model = treelite.frontend.from_xgboost(regressor.get_booster())  # type: ignore[union-attr]
        tl2cgen.export_lib(tl_model, toolchain=toolchain, libpath=libpath, verbose=False)  # type: ignore[union-attr]

    predictor = tl2cgen.Predictor(libpath, nthread=os.cpu_count(), verbose=False)  # type: ignore[union-attr]
    return TL2CGenXGBPerTupleModel(predictor)


def train_xgboost_per_tuple_model(predicted_cardinalities: bool = False) -> XGBPerTupleModel:
    if not _HAS_XGBOOST:
        raise RuntimeError(
            "XGBoost is not installed. Install it with: pip install xgboost\n"
            "(then rerun: python -m src.compare_xgboost)"
        )

    feature_mapper = FeatureMapper()
    train_benchmarks = DataCollector.collect_benchmarks(DatabaseManager.get_train_databases(), predicted_cardinalities)
    x, y = _build_per_tuple_training_data(train_benchmarks, feature_mapper)

    seed = 21
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)

    reg = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        tree_method="approx",
        grow_policy="lossguide",
        max_leaves=31,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=1.0,
        objective="reg:squarederror",
        eval_metric="mape",
        n_jobs=-1,
        random_state=seed,
    )
    reg.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return XGBPerTupleModel(reg)


def main():
    predicted_cardinalities = False

    run_compiled = os.environ.get("T3_RUN_COMPILED", "0") == "1"
    run_xgb_compiled = os.environ.get("T3_RUN_XGB_COMPILED", "0") == "1"

    print("--- Comparing: Original T3 vs XGBoost (same data pipeline) ---")

    start = time.perf_counter()
    t3_model = optimize_all(predicted_cardinalities)
    t3_train_s = time.perf_counter() - start

    feature_mapper = FeatureMapper()
    train_benchmarks = DataCollector.collect_benchmarks(DatabaseManager.get_train_databases(), predicted_cardinalities)
    bench_x, _bench_y = _build_per_tuple_training_data(train_benchmarks, feature_mapper)

    t3_table, t3_inf_ms = _evaluate_model(t3_model, predicted_cardinalities)

    print("\nT3 (original) — Q-Error table")
    print(t3_table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"[Timing] Train: {t3_train_s:.2f}s | Avg inference: {t3_inf_ms:.3f} ms/query")
    t3_us_row = _benchmark_batch_predict(lambda m: t3_model.tree.predict(m), bench_x)
    print(f"[Model-only] LightGBM predict: {t3_us_row:.3f} us/row")

    if run_compiled:
        if _HAS_LLEAVES:
            start = time.perf_counter()
            t3_compiled = compile_t3_with_lleaves(t3_model, Path("compare_output") / "compiled")
            t3_compile_s = time.perf_counter() - start

            t3c_table, t3c_inf_ms = _evaluate_model(t3_compiled, predicted_cardinalities)
            print("\nT3 (lleaves compiled) — Q-Error table")
            print(t3c_table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
            print(
                f"[Timing] Compile: {t3_compile_s:.2f}s | Avg inference: {t3c_inf_ms:.3f} ms/query"
            )
            t3c_us_row = _benchmark_batch_predict(lambda m: t3_compiled.compiled_model.predict(m), bench_x)
            speedup = t3_us_row / max(1e-12, t3c_us_row)
            print(
                f"[Model-only] lleaves predict: {t3c_us_row:.3f} us/row (x{speedup:.2f} speedup)"
            )
        else:
            print("\n[WARN] lleaves not installed — skipping compiled T3 evaluation.")

    if _HAS_XGBOOST:
        start = time.perf_counter()
        xgb_model = train_xgboost_per_tuple_model(predicted_cardinalities)
        xgb_train_s = time.perf_counter() - start

        xgb_table, xgb_inf_ms = _evaluate_model(xgb_model, predicted_cardinalities)
        print("\nXGBoost (per-tuple) — Q-Error table")
        print(xgb_table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print(f"[Timing] Train: {xgb_train_s:.2f}s | Avg inference: {xgb_inf_ms:.3f} ms/query")

        xgb_us_row = _benchmark_batch_predict(lambda m: xgb_model.regressor.predict(m), bench_x)
        print(f"[Model-only] XGBoost predict: {xgb_us_row:.3f} us/row")

        if run_xgb_compiled:
            if _HAS_TL2CGEN:
                start = time.perf_counter()
                xgb_compiled = compile_xgboost_with_tl2cgen(
                    xgb_model.regressor,
                    Path("compare_output") / "compiled",
                )
                xgb_compile_s = time.perf_counter() - start

                xgbc_table, xgbc_inf_ms = _evaluate_model(xgb_compiled, predicted_cardinalities)
                print("\nXGBoost (tl2cgen compiled) — Q-Error table")
                print(xgbc_table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
                print(
                    f"[Timing] Compile: {xgb_compile_s:.2f}s | Avg inference: {xgbc_inf_ms:.3f} ms/query"
                )

                bench_dmat = tl2cgen.DMatrix(bench_x)  # type: ignore[union-attr]
                xgbc_us_row = _benchmark_batch_predict_fixed(
                    lambda: xgb_compiled.predictor.predict(bench_dmat),
                    n_rows=int(bench_x.shape[0]),
                )
                speedup = xgb_us_row / max(1e-12, xgbc_us_row)
                print(
                    f"[Model-only] tl2cgen predict: {xgbc_us_row:.3f} us/row (x{speedup:.2f} speedup)"
                )
            else:
                print("\n[WARN] tl2cgen/treelite not available — skipping compiled XGBoost evaluation.")
    else:
        print("\n[WARN] XGBoost not installed — skipping XGBoost comparison.")
        print("       Install with: pip install xgboost")


if __name__ == "__main__":
    main()