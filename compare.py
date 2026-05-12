from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.features import FeatureMapper
from src.metrics import abs_error, q_error
from src.train import optimize_all

from src.compare_utils import (
    benchmark_batch_predict as _benchmark_batch_predict,
    benchmark_batch_predict_fixed as _benchmark_batch_predict_fixed,
    build_accuracy_slices as _build_accuracy_slices,
)

from src.training_data import build_per_tuple_training_data as _build_per_tuple_training_data

from src.compare_xgboost import (
    _HAS_LLEAVES,
    _HAS_TL2CGEN,
    _HAS_XGBOOST,
    compile_t3_with_lleaves,
    compile_xgboost_with_tl2cgen,
    train_xgboost_per_tuple_model,
)

from src.compare_catboost import (
    _HAS_CATBOOST,
    compile_catboost_to_cpp,
    train_catboost_per_tuple_model,
)


try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None  # type: ignore


try:
    import xgboost  # type: ignore
except Exception:
    xgboost = None  # type: ignore

try:
    import catboost  # type: ignore
except Exception:
    catboost = None  # type: ignore


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _quantiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"n": 0, "avg": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "n": int(values.size),
        "avg": float(np.mean(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _model_name(model: Any) -> str:
    name = model.__class__.__name__
    if name == "PerTupleTreeModel":
        return "T3 (LightGBM PerTupleTreeModel)"
    return name


def _evaluate_model_on_queries(model: Any, queries: list[Any]) -> tuple[np.ndarray, float]:
    qerrs: list[float] = []
    start = time.perf_counter()
    for q in queries:
        true_s = float(q.get_total_runtime())
        pred_s = float(model.estimate_runtime(q))
        qerrs.append(float(q_error(true_s, pred_s)))
    elapsed = time.perf_counter() - start
    avg_ms_per_query = (elapsed / max(1, len(queries))) * 1000.0
    return np.array(qerrs, dtype=float), avg_ms_per_query


def _iter_per_query_rows(model: Any, model_label: str, queries: Iterable[Any]):
    for q in queries:
        true_s = float(q.get_total_runtime())
        pred_s = float(model.estimate_runtime(q))
        qerr = float(q_error(true_s, pred_s))
        aerr = float(abs_error(true_s, pred_s))

        db = getattr(q.query_plan, "db", None)
        db_schema = getattr(getattr(db, "schema", None), "name", None)
        db_path = getattr(db, "name", None)
        if hasattr(db, "get_path"):
            try:
                db_path = db.get_path()
            except Exception:
                pass

        yield {
            "model": model_label,
            "dataset": "all",
            "query_name": q.name,
            "query_category": getattr(q.query_category, "name", str(q.query_category)),
            "db_schema": db_schema,
            "db_path": db_path,
            "true_s": true_s,
            "pred_s": pred_s,
            "q_error": qerr,
            "abs_error_s": aerr,
            "log10_q_error": float(np.log10(max(qerr, 1e-12))),
        }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _write_markdown_worst(path: Path, rows: list[dict[str, Any]], model_label: str, topk: int) -> None:
    rows_sorted = sorted(rows, key=lambda r: float(r["q_error"]), reverse=True)[:topk]

    lines = []
    lines.append(f"# Worst queries — {model_label}")
    lines.append("")
    lines.append(f"Top {len(rows_sorted)} by Q-Error.")
    lines.append("")
    lines.append("| rank | q_error | true_s | pred_s | abs_error_s | db_schema | category | query_name |")
    lines.append("|---:|---:|---:|---:|---:|---|---|---|")
    for i, r in enumerate(rows_sorted, 1):
        lines.append(
            "| {rank} | {qerr:.3f} | {true_s:.6f} | {pred_s:.6f} | {aerr:.6f} | {db_schema} | {cat} | {name} |".format(
                rank=i,
                qerr=float(r["q_error"]),
                true_s=float(r["true_s"]),
                pred_s=float(r["pred_s"]),
                aerr=float(r["abs_error_s"]),
                db_schema=r.get("db_schema") or "",
                cat=r.get("query_category") or "",
                name=r.get("query_name") or "",
            )
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _env_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "numpy": getattr(np, "__version__", None),
        "lightgbm": getattr(lgb, "__version__", None) if lgb is not None else None,
        "xgboost": getattr(xgboost, "__version__", None) if xgboost is not None else None,
        "catboost": getattr(catboost, "__version__", None) if catboost is not None else None,
    }

def main() -> int:
    parser = argparse.ArgumentParser(description="Compare T3 vs XGBoost vs CatBoost")
    parser.add_argument("--outdir", default="compare_output", help="Base output directory")
    parser.add_argument("--topk", type=int, default=50, help="How many worst queries to list per model")
    parser.add_argument(
        "--predicted-cardinalities",
        action="store_true",
        help="Use predicted (instead of analyzed) cardinalities when parsing plans",
    )
    args = parser.parse_args()

    predicted_cardinalities = bool(args.predicted_cardinalities)
    base_outdir = Path(args.outdir)
    run_dir = base_outdir / f"run_{_now_stamp()}"
    _safe_mkdir(run_dir)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "env": _env_metadata(),
    }
    _write_json(run_dir / "meta.json", meta)

    # Pre-build benchmark matrix for model-only latency tests
    feature_mapper = FeatureMapper()
    train_benchmarks = DataCollector.collect_benchmarks(DatabaseManager.get_train_databases(), predicted_cardinalities)
    bench_x, _ = _build_per_tuple_training_data(train_benchmarks, feature_mapper)
    bench_x_contig = np.ascontiguousarray(bench_x, dtype=np.float32)
    n_rows, n_cols = bench_x_contig.shape
    bench_out = np.zeros(n_rows, dtype=np.float32)

    models: list[tuple[str, Any, float]] = []

    # 1. T3 Baseline
    print("=== Training original T3 model ===")
    t0 = time.perf_counter()
    t3_model = optimize_all(predicted_cardinalities)
    t3_train_s = time.perf_counter() - t0
    models.append(("T3 (original)", t3_model, t3_train_s))

    if _HAS_LLEAVES:
        print("=== Compiling T3 (lleaves) ===")
        t0 = time.perf_counter()
        t3_compiled = compile_t3_with_lleaves(t3_model, run_dir / "compiled")
        compile_s = time.perf_counter() - t0
        models.append(("T3 (compiled lleaves)", t3_compiled, t3_train_s + compile_s))

    # 2. XGBoost
    if _HAS_XGBOOST:
        print("=== Training XGBoost ===")
        t0 = time.perf_counter()
        xgb_model = train_xgboost_per_tuple_model(predicted_cardinalities)
        xgb_train_s = time.perf_counter() - t0
        models.append(("XGBoost (uncompiled)", xgb_model, xgb_train_s))

        if _HAS_TL2CGEN:
            print("=== Compiling XGBoost (tl2cgen) ===")
            t0 = time.perf_counter()
            xgb_compiled = compile_xgboost_with_tl2cgen(xgb_model.regressor, run_dir / "compiled")
            compile_s = time.perf_counter() - t0
            models.append(("XGBoost (compiled tl2cgen)", xgb_compiled, xgb_train_s + compile_s))

    # 3. CatBoost
    if _HAS_CATBOOST:
            print("=== Training CatBoost ===")
            t0 = time.perf_counter()
            cat_model = train_catboost_per_tuple_model(predicted_cardinalities)
            cat_train_s = time.perf_counter() - t0
            models.append(("CatBoost (uncompiled)", cat_model, cat_train_s))

            print("=== Compiling CatBoost (C++) ===")
            t0 = time.perf_counter()
            cat_compiled = compile_catboost_to_cpp(cat_model, run_dir / "compiled")
            compile_s = time.perf_counter() - t0
            models.append(("CatBoost (compiled C++)", cat_compiled, cat_train_s + compile_s))

    # Prepare datasets
    paper_slices = _build_accuracy_slices(predicted_cardinalities)
    all_queries_name = "All Queries (All DBs)"
    all_queries = DataCollector.collect_benchmarks(DatabaseManager.get_all_databases(), predicted_cardinalities)

    # Evaluate and save summary
    print(f"\\n=== Evaluating {len(models)} models ===")
    summary_rows: list[dict[str, Any]] = []

    per_query_rows: list[dict[str, Any]] = []

    for model_label, model, train_s in models:
        # Evaluate model on slices
        rows_for_print = []
        for dataset_name, queries in (paper_slices + [(all_queries_name, all_queries)]):
            qerrs, avg_ms = _evaluate_model_on_queries(model, queries)
            stats = _quantiles(qerrs)
            
            if dataset_name != all_queries_name:
                rows_for_print.append({
                    "Dataset": dataset_name,
                    "p50": stats["p50"],
                    "p90": stats["p90"],
                    "Avg": stats["avg"]
                })
            
            summary_rows.append(
                {
                    "model": model_label,
                    "dataset": dataset_name,
                    "train_s": float(train_s),
                    "avg_inference_ms_per_query": float(avg_ms),
                    **stats,
                }
            )

        print(f"\\n{model_label} — Q-Error table")
        df_print = pd.DataFrame(rows_for_print)
        print(df_print.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

        # Calculate end-to-end latency across all queries
        start = time.perf_counter()
        for q in all_queries:
            _ = model.estimate_runtime(q)
        elapsed = time.perf_counter() - start
        avg_ms_e2e = (elapsed / max(1, len(all_queries))) * 1000.0

        print(f"[Timing] Train/Compile: {train_s:.2f}s | Avg inference: {avg_ms_e2e:.3f} ms/query")

        # Calculate model-only latency
        us_row = None
        try:
            if "compiled lleaves" in model_label:
                us_row = _benchmark_batch_predict(lambda m: model.compiled_model.predict(m), bench_x)
            elif "T3 (original)" in model_label:
                us_row = _benchmark_batch_predict(lambda m: model.tree.predict(m), bench_x)
            elif "XGBoost (uncompiled)" in model_label:
                us_row = _benchmark_batch_predict(lambda m: model.regressor.predict(m), bench_x)
            elif "XGBoost (compiled tl2cgen)" in model_label:
                import tl2cgen
                bench_dmat = tl2cgen.DMatrix(bench_x)
                us_row = _benchmark_batch_predict_fixed(lambda: model.predictor.predict(bench_dmat), n_rows=n_rows)
            elif "CatBoost (uncompiled)" in model_label:
                us_row = _benchmark_batch_predict(lambda m: model.regressor.predict(m), bench_x)
            elif "CatBoost (compiled C++)" in model_label:
                us_row = _benchmark_batch_predict_fixed(lambda: model.lib.predict_batch(bench_x_contig, bench_out, n_rows, n_cols), n_rows=n_rows)
            
            if us_row is not None:
                print(f"[Model-only] Predict: {us_row:.3f} us/row")
        except Exception as e:
            print(f"[Model-only] Predict failed: {e}")

        per_query_rows_for_model = list(_iter_per_query_rows(model, model_label, all_queries))
        per_query_rows.extend(per_query_rows_for_model)

        clean_label = model_label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p")
        _write_markdown_worst(run_dir / f"worst_{clean_label}.md", per_query_rows_for_model, model_label, args.topk)

    summary_fields = [
        "model",
        "dataset",
        "train_s",
        "avg_inference_ms_per_query",
        "n",
        "avg",
        "p10",
        "p50",
        "p90",
        "p95",
        "max",
    ]
    _write_csv(run_dir / "summary.csv", summary_rows, summary_fields)

    per_query_fields = [
        "model",
        "query_name",
        "query_category",
        "db_schema",
        "db_path",
        "true_s",
        "pred_s",
        "q_error",
        "abs_error_s",
        "log10_q_error",
    ]
    _write_csv(run_dir / "per_query.csv", per_query_rows, per_query_fields)

    # Save a compact human-readable report
    # print(f"\\n=== Writing report.md to {run_dir} ===")
    report_lines = []
    report_lines.append("# Comparison Report (T3 vs XGBoost vs CatBoost)")
    report_lines.append("")
    report_lines.append(f"Output folder: `{run_dir}`")
    report_lines.append("")
    report_lines.append("## Artifacts")
    report_lines.append("")
    report_lines.append("- `meta.json`: environment + command args")
    report_lines.append("- `summary.csv`: per-dataset quantiles (p10/p50/p90/p95/max) + timing")
    report_lines.append("- `per_query.csv`: per-query true/pred/q-error for deep analysis")
    report_lines.append("- `worst_*.md`: top-k worst queries per model")
    report_lines.append("")
    report_lines.append("## Quick view")
    report_lines.append("")

    summary_sorted = sorted(summary_rows, key=lambda r: (r["dataset"], r["model"]))
    report_lines.append("| dataset | model | p50 | p90 | avg | max | avg_inference_ms | train_s | n |")
    report_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary_sorted:
        report_lines.append(
            "| {dataset} | {model} | {p50:.2f} | {p90:.2f} | {avg:.2f} | {maxv:.2f} | {inf:.3f} | {train:.2f} | {n} |".format(
                dataset=r["dataset"],
                model=r["model"],
                p50=float(r["p50"]),
                p90=float(r["p90"]),
                avg=float(r["avg"]),
                maxv=float(r["max"]),
                inf=float(r["avg_inference_ms_per_query"]),
                train=float(r["train_s"]),
                n=int(r["n"]),
            )
        )

    with open(run_dir / "report.md", "w") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
