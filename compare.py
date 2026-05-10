#!/usr/bin/env python3

"""Compare and save evaluation results for T3 vs an XGBoost variant.

This is a *report generator* meant to make results easier to interpret than console-only output.

What it does:
- Trains the original T3 model (LightGBM) using the repo's normal pipeline.
- Optionally trains an XGBoost variant on the *same* per-tuple pipeline dataset.
- Evaluates both models on:
  - the same slices used by the paper's accuracy table
  - the full benchmark corpus (all DBs)
- Saves detailed per-query results and summary statistics to a timestamped folder.

Run (from repo root):

  source venv/bin/activate
  python compare.py

Optional:

  python compare.py --topk 100 --outdir compare_output

Notes:
- If XGBoost isn't installed, the script will still run the T3 baseline and note that XGBoost was skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.metrics import abs_error, q_error
from src.optimizer import QueryCategory
from src.train import optimize_all


try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None  # type: ignore


try:
    import xgboost  # type: ignore

    _HAS_XGBOOST = True
except Exception:
    xgboost = None  # type: ignore
    _HAS_XGBOOST = False


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetSlice:
    name: str
    queries: list[Any]  # BenchmarkedQuery, but avoid import cycle typing


def _paper_slices(predicted_cardinalities: bool) -> list[DatasetSlice]:
    return [
        DatasetSlice(
            "Train Queries",
            DataCollector.collect_benchmarks(DatabaseManager.get_train_databases(), predicted_cardinalities),
        ),
        DatasetSlice(
            "All TPC-DS Test Queries",
            DataCollector.collect_benchmarks(DatabaseManager.get_test_databases(), predicted_cardinalities),
        ),
        DatasetSlice(
            "TPC-DS Benchmark Queries",
            DataCollector.collect_benchmarks(
                DatabaseManager.get_test_databases(),
                predicted_cardinalities,
                query_category=[QueryCategory.fixed],
            ),
        ),
        DatasetSlice(
            "TPC-DS sf 100 Test Queries",
            DataCollector.collect_benchmarks([DatabaseManager.get_database("tpcdsSf100")], predicted_cardinalities),
        ),
        DatasetSlice(
            "TPC-DS sf 100 Benchmark Queries",
            DataCollector.collect_benchmarks(
                [DatabaseManager.get_database("tpcdsSf100")],
                predicted_cardinalities,
                query_category=[QueryCategory.fixed],
            ),
        ),
    ]


def _all_queries_slice(predicted_cardinalities: bool) -> DatasetSlice:
    return DatasetSlice(
        "All Queries (All DBs)",
        DataCollector.collect_benchmarks(DatabaseManager.get_all_databases(), predicted_cardinalities),
    )


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
    # Heuristic naming for common models in this repo
    if name == "PerTupleTreeModel":
        return "T3 (LightGBM PerTupleTreeModel)"
    return name


def _evaluate_model_on_slice(model: Any, dataset: DatasetSlice) -> tuple[np.ndarray, float]:
    qerrs: list[float] = []
    start = time.perf_counter()
    for q in dataset.queries:
        true_s = float(q.get_total_runtime())
        pred_s = float(model.estimate_runtime(q))
        qerrs.append(float(q_error(true_s, pred_s)))
    elapsed = time.perf_counter() - start
    avg_ms_per_query = (elapsed / max(1, len(dataset.queries))) * 1000.0
    return np.array(qerrs, dtype=float), avg_ms_per_query


def _iter_per_query_rows(model: Any, model_label: str, queries: Iterable[Any]):
    for q in queries:
        true_s = float(q.get_total_runtime())
        pred_s = float(model.estimate_runtime(q))
        qerr = float(q_error(true_s, pred_s))
        aerr = float(abs_error(true_s, pred_s))

        # Extra context for interpretation
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


def _train_xgboost_model(predicted_cardinalities: bool):
    # Reuse the implementation that matches T3's per-tuple target transform.
    from src.compare_xgboost import train_xgboost_per_tuple_model

    return train_xgboost_per_tuple_model(predicted_cardinalities)


def _env_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "numpy": getattr(np, "__version__", None),
        "lightgbm": getattr(lgb, "__version__", None) if lgb is not None else None,
        "xgboost": getattr(xgboost, "__version__", None) if xgboost is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare T3 vs XGBoost and save detailed results")
    parser.add_argument("--outdir", default="compare_output", help="Base output directory")
    parser.add_argument("--topk", type=int, default=50, help="How many worst queries to list per model")
    parser.add_argument(
        "--predicted-cardinalities",
        action="store_true",
        help="Use predicted (instead of analyzed) cardinalities when parsing plans",
    )
    parser.add_argument(
        "--skip-xgboost",
        action="store_true",
        help="Skip training/evaluating XGBoost even if installed",
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

    # Train baseline model
    print("[1/4] Training original T3 model...")
    t0 = time.perf_counter()
    t3_model = optimize_all(predicted_cardinalities)
    t3_train_s = time.perf_counter() - t0
    t3_label = _model_name(t3_model)

    models: list[tuple[str, Any, float]] = [(t3_label, t3_model, t3_train_s)]

    # Train XGBoost model (optional)
    if args.skip_xgboost:
        print("[2/4] Skipping XGBoost (flag set)")
    elif not _HAS_XGBOOST:
        print("[2/4] XGBoost not installed; skipping XGBoost")
    else:
        print("[2/4] Training XGBoost per-tuple model...")
        t0 = time.perf_counter()
        xgb_model = _train_xgboost_model(predicted_cardinalities)
        xgb_train_s = time.perf_counter() - t0
        models.append(("XGBoost (per-tuple)", xgb_model, xgb_train_s))

    # Prepare datasets
    slices = _paper_slices(predicted_cardinalities) + [_all_queries_slice(predicted_cardinalities)]

    # Evaluate and save summary
    print("[3/4] Evaluating models and writing summaries...")
    summary_rows: list[dict[str, Any]] = []

    # Per-query results across all queries (for detailed inspection)
    all_queries = _all_queries_slice(predicted_cardinalities).queries
    per_query_rows: list[dict[str, Any]] = []

    for model_label, model, train_s in models:
        for ds in slices:
            qerrs, avg_ms = _evaluate_model_on_slice(model, ds)
            stats = _quantiles(qerrs)
            summary_rows.append(
                {
                    "model": model_label,
                    "dataset": ds.name,
                    "train_s": float(train_s),
                    "avg_inference_ms_per_query": float(avg_ms),
                    **stats,
                }
            )

        # Detailed per-query rows
        per_query_rows_for_model = list(_iter_per_query_rows(model, model_label, all_queries))
        per_query_rows.extend(per_query_rows_for_model)

        # Worst offenders markdown
        _write_markdown_worst(run_dir / f"worst_{model_label.replace(' ', '_').replace('(', '').replace(')', '')}.md", per_query_rows_for_model, model_label, args.topk)

    # Write summary + per-query CSV
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

    # Also save a compact human-readable report
    print("[4/4] Writing report.md...")
    report_lines = []
    report_lines.append("# Comparison Report")
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

    # Embed a small table (sorted for readability)
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

    print(f"Done. Results saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
