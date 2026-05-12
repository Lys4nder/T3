# Context / work log

This file records what was checked in the repo and what sources were used as ground truth for the corrections.

## Scope
- Goal: validate dissertation claims and bibliography against the PDFs in `documentation/` and against the repo’s actual experimental outputs.
- Focus: numeric/data correctness, citation integrity (year/venue/DOI), and whether the cited PDFs match what the BibTeX claims.

## What was scanned

### LaTeX dissertation sources
- `documentation/Dissertation Lysander Pitu/main.tex`
- `documentation/Dissertation Lysander Pitu/chapters/chapter1_introduction.tex`
- `documentation/Dissertation Lysander Pitu/chapters/chapter2.tex`
- `documentation/Dissertation Lysander Pitu/chapters/chapter3.tex`
- `documentation/Dissertation Lysander Pitu/chapters/chapter4.tex`
- `documentation/Dissertation Lysander Pitu/chapters/chapter6_conclusions.tex`
- `documentation/Dissertation Lysander Pitu/references.bib`

### PDF corpus (primary studies)
- Folder scanned: `documentation/02_SLR 2/` (30 PDFs, numbered `1.` … `30.`)
- Note: there is a second folder `documentation/OneDrive Practical Part 04/` that appears to duplicate the same numbered PDFs.

## Tools/method
- PDF parsing: Python `pypdf` (`PdfReader`) to extract basic metadata and first-page text.
- Spot checks: searched for DBMS names (e.g., “PostgreSQL”, “Umbra”) in the first ~10 pages of the T3 PDF.

## Key findings (evidence)

### T3 paper: DBMS name
- Source: `documentation/02_SLR 2/23. ...T3...pdf`
- Result: early pages contain “Umbra” and do *not* contain “PostgreSQL”.
- Evidence: page 4 states that T3 predicts the execution time of **Umbra** and that concepts may transfer to other systems.

### PerfCE paper: PDF mismatch
- Source: `documentation/02_SLR 2/21. PerfCE_...pdf`
- Result: first page header reads “JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015”, which strongly indicates a template rather than a real venue PDF.
- Implication: the BibTeX entry `Ji2024` cannot be treated as verified against the provided PDF.

### BibTeX integrity problems
- Source: `documentation/Dissertation Lysander Pitu/references.bib`
- Observed issues:
  - A line starting with `#` (not a BibTeX comment) likely breaks `bibtex`.
  - `@misc{MethamorphicTesting2026,...}` is malformed (empty `key`, missing proper URL/title/urldate).
  - `Ji2024` key/year mismatch (`Ji2024` but `year={2023}`) and placeholder journal.

## Experimental outputs used as ground truth
- Comparison outputs referenced in the conversation:
  - `compare_output/run_20260505_193206/` (contains `report.md`, `summary.csv`, `per_query.csv`, and metadata)
- Intended use in thesis:
  - Copy the reported q-error statistics (p50/p90/avg/max) and timing numbers into Chapter 4.
  - Record the exact environment (machine specs + versions) used for the run.

## XGBoost extension + alignment experiments (repo work)

### What changed in the code
- File: `src/compare_xgboost.py`
- Implemented “Change 1 + 2” from the alignment note:
  - Track MAPE during training (`eval_metric="mape"`).
  - Use leaf-limited trees to approximate “~30 leaves per tree” (`grow_policy="lossguide"`, `max_leaves=31`, `max_depth=0`, `tree_method="hist"`).
  - Added a small clamp after the `-log(t)` transform to keep labels positive (`y >= 1e-6`) so MAPE is well-defined.

### Previous vs new results (XGBoost)
Two consecutive runs were compared:

**Previous (before Change 1+2)**
- Train Queries: p50 1.16, p90 1.61, Avg 2.30
- All TPC-DS Test Queries: p50 1.24, p90 2.15, Avg 1.90
- TPC-DS Benchmark Queries: p50 1.35, p90 3.05, Avg 1.95
- TPC-DS sf 100 Test Queries: p50 1.31, p90 2.53, Avg 2.00
- TPC-DS sf 100 Benchmark Queries: p50 1.43, p90 3.52, Avg 2.15
- Timing: Train 1.28s | Avg inference 0.251 ms/query

**New (after Change 1+2)**
- Train Queries: p50 1.20, p90 1.83, Avg 2.25
- All TPC-DS Test Queries: p50 1.25, p90 2.29, Avg 1.84
- TPC-DS Benchmark Queries: p50 1.35, p90 2.92, Avg 1.83
- TPC-DS sf 100 Test Queries: p50 1.29, p90 2.77, Avg 2.00
- TPC-DS sf 100 Benchmark Queries: p50 1.47, p90 3.14, Avg 1.98
- Timing: Train 1.88s | Avg inference 0.223 ms/query

Interpretation (high level): average q-error improved on most slices, but p90 got worse on several; training got slower.

## Compiled inference work (LightGBM / T3)

### What was implemented
- File: `src/compare_xgboost.py`
- Added an optional “compiled T3” evaluation path using the Python `lleaves` package.
- Enabled via environment variable: `T3_RUN_COMPILED=1`.
- This produces:
  - the usual end-to-end “ms/query” timing (includes Python feature extraction)
  - a **model-only** benchmark (batch prediction on a prebuilt matrix) that isolates predictor speed

### Key caveat for dissertation claims
End-to-end “ms/query” numbers are dominated by Python overhead (feature extraction + loop), so compilation can appear slower even when the predictor is faster.

For any “compiled model is faster” claim, use the model-only benchmark (or move feature extraction out of the critical path) and clearly label which timing is reported.

### Example observed effect
In one run, model-only prediction showed an ~8× speedup for `lleaves` vs interpreted LightGBM prediction.

## Compiled XGBoost inference (tl2cgen + Treelite)
- File: `src/compare_xgboost.py`
- Implemented compiled inference for the XGBoost per-tuple model using `tl2cgen`.
- Enabled via environment variable: `T3_RUN_XGB_COMPILED=1`.
- Like the `lleaves` path, this produces both:
  - end-to-end “ms/query” (includes Python feature extraction)
  - a **model-only** benchmark (us/row) to isolate predictor speed

### macOS notes (OpenMP)
- `treelite` / `tl2cgen` require the OpenMP runtime on macOS (`libomp.dylib`).
- Working setup:
  - `brew install libomp`
  - `DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH`

### Example observed effect (one run)
- Uncompiled XGBoost model-only: ~0.737 us/row
- tl2cgen compiled model-only: ~0.215 us/row (~3.43×)
- End-to-end avg inference (includes Python overhead): 0.248 ms/query → 0.070 ms/query in that run

### Known warning during compile
- During compilation, tl2cgen emitted a warning about a Treelite version mismatch (runtime reported 4.1.2 vs checkpoint generated by 4.7.0) and “Parallel compilation disabled”.
- Despite the warning, compiled and uncompiled XGBoost produced identical q-error tables in the validation run.
- For dissertation/reproducibility: treat this as a dependency/toolchain detail to document and (optionally) eliminate by pinning versions and running on a clean Linux toolchain.

## Compiled CatBoost inference (Native C++ / ctypes)
- File: `src/compare_catboost.py`
- Implemented a native C++ compilation pipeline for CatBoost:
  - Saved model as standalone C++ (`format="CPP"`). Note: Required changing `grow_policy` to `SymmetricTree` since CatBoost C++ export only supports symmetric trees.
  - Generated a C++ wrapper to batch-process predictions.
  - Compiled directly into a shared object (`.dylib` / `.so`) via `clang++` / `g++`.
  - Invoked the compiled function via Python's `ctypes` for near-zero overhead inference.
- Example observed effect:
  - **Massive End-to-End Speedup**: The compiled CatBoost variant reduced average E2E latency to ~0.018 ms/query (an impressive ~78% reduction compared to original T3's 0.082 ms/query), completely bypassing the Python-to-C++ wrapper overhead of the default CatBoost library.
  - Pure model-only speed (uncompiled native CatBoost engine) is incredibly fast (~0.235 us/row), beating standard compiled tree speeds due to deep SIMD optimizations, but its Python wrapper heavily penalizes E2E latency if not compiled out via our pipeline.

## Unified Model Comparison (`compare.py`)
- Updated `compare.py` to evaluate all 6 variants in a single run (if dependencies are available):
  - T3 Baseline & T3 Compiled (`lleaves`)
  - XGBoost & XGBoost Compiled (`tl2cgen`)
  - CatBoost & CatBoost Compiled (Native C++)
- Added direct terminal output for `pandas`-formatted Q-Error tables and latency benchmarks (`[Model-only]` vs `[Timing] E2E`), perfectly mirroring the output of the individual scripts.
- This serves as the primary data generator for the Chapter 4 benchmarking tables.

## Docker/Linux reproducibility (recommended for compilation)
- Docker image `t3-compile-xgb` builds successfully and installs the pinned packages from `requirements.txt`.
- Note: the `Dockerfile` pins `--platform=linux/amd64`. On Apple Silicon (arm64) hosts, Docker will typically run this image under emulation and will print a platform mismatch warning; compilation may be slower.
- Note: the current `Dockerfile` only installs dependencies; it does not copy the repo. Run with a bind mount:

  - Build: `docker build -t t3-compile-xgb .`
  - Run (example):
    - `docker run --rm -it \
        -v "$PWD":/app -w /app \
        --entrypoint /venv/bin/python \
        t3-compile-xgb -m src.compare_xgboost`
    - Compiled XGBoost: add `-e T3_RUN_XGB_COMPILED=1`

## Open items / what’s missing to fully “close the loop”
- PerfCE: obtain the *real* PerfCE paper PDF (or arXiv/DOI) and update `references.bib` + cites accordingly.
- Results chapter: populate `chapter4.tex` with the actual experimental tables/figures derived from `compare_output/...`.
- PRISMA/selection: explicitly state the numeric selection flow in Chapter 2 and ensure it matches your raw SLR dataset.

### Dissertation-facing suggestion
Keep this file, but treat it as a **lab notebook / provenance log**, not something to cite verbatim.

For the dissertation itself, extract only the defensible, reproducible pieces:
- “What the system does” (pipeline features, per-tuple target, $t'=-\log(t)$, scan-size scaling).
- “What we changed” (XGBoost learner swap, leaf-limited config, eval metric).
- “What we measured” (q-error tables + timing), with strict wording about what timings include.
- “What is compiled” (compiled LightGBM predictor via `lleaves`) and how it is benchmarked (model-only vs end-to-end).

If you include compilation results in the dissertation, the cleanest structure is:
- Main text: accuracy tables and one short latency table.
- Appendix: exact run commands + environment notes (including any `DYLD_LIBRARY_PATH` requirement).
