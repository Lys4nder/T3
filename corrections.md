# Corrections (data + consistency)

This file lists concrete corrections to make the dissertation *factually consistent* with the PDFs in `documentation/` and with the actual experimental outputs produced in this repo.

## Critical (must fix)

### 1) T3 target DBMS is **Umbra**, not PostgreSQL
- **What’s wrong:** Any claim that T3 “predicts latency in PostgreSQL” is not supported by the T3 paper as provided in `documentation/`.
- **Evidence (PDF):** In the T3 paper (page 4, “Problem Scope”), the text states: “T3 predicts the execution time of **Umbra**, a flash-based compiling relational database system…”.
- **Correction:**
  - If the dissertation text currently mentions PostgreSQL as the system targeted by T3, change it to **Umbra**.
  - If your work *adapts* T3 ideas to PostgreSQL in your implementation, state that explicitly as *your extension* and avoid attributing it to the original T3 paper.

### 2) PerfCE PDF appears to be a LaTeX journal template (bibliography/citation mismatch)
- **What’s wrong:** The PDF file used for the PerfCE reference does not look like a peer‑reviewed publication PDF; it looks like a template (“JOURNAL OF LATEX CLASS FILES…”). This makes the bibliography entry `Ji2024` unreliable as currently written.
- **Evidence (PDF):** The first page header of the PerfCE PDF is: “JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015”.
- **Evidence (BibTeX):** `references.bib` contains:
  - key `Ji2024`, but `year = {2023}`
  - `journal = {Journal of LaTeX Class Files}` (which aligns with the template header, not a real venue)
- **Correction (pick one):**
  1) Replace the PDF with the *actual PerfCE paper PDF* (or its arXiv version), then update `Ji2024` to match (authors, year, venue, DOI/arXiv).
  2) If you cannot locate the real paper, remove PerfCE from the set of “peer‑reviewed” primary studies and treat it as “unverified / excluded”, updating Chapter 2 accordingly.
- **Also required:** Make the citation key/year consistent (`Ji2024` should not have `year={2023}` unless you rename the key and all cites).

### 3) `references.bib` currently contains BibTeX-breaking syntax
- **What’s wrong:** The bibliography file contains a line starting with `#`, which is not a BibTeX comment character. This can break `bibtex`.
- **Evidence:**
  - `#nu stiu daca asa e corect sa citez wikipedia`
- **Correction:** Replace `#...` with a BibTeX-safe comment (e.g., `% ...`) or `@comment{...}`.

### 4) Wikipedia “metamorphic testing” citation is invalid/incomplete
- **What’s wrong:** The entry is malformed and likely won’t compile:
  - `key = ,` is invalid
  - `note = https://...` is missing braces/quotes and lacks title/urldate
- **Correction:** Replace with a proper `@misc` (or remove it entirely if you can cite a peer‑reviewed source for the definition).

## Major (should fix)

### 5) Chapter 2 inclusion criterion “peer-reviewed venue” conflicts with “Preprint” entries
- **What’s wrong:** Chapter 2 says papers are from recognized peer‑reviewed venues, but `references.bib` marks several items as `Preprint` / `arXiv preprint`.
- **Correction options:**
  - Relax the criterion to explicitly allow *preprints* (and define how you treat them), OR
  - Exclude preprints from the final primary set, OR
  - Replace preprints with their final published versions if available.

### 6) PRISMA/selection counts (“180 → 30”) are not present in the LaTeX
- **What’s wrong:** The methodology text mentions selection and duplicates removal, but the concrete numeric flow (the “180 initial → 30 final” mentioned in `instructions.md`) isn’t currently stated in the dissertation LaTeX.
- **Correction:** Add explicit counts per stage and include a PRISMA-style diagram (or a clear selection-flow figure) matching those counts.

### 7) Experimental results chapter is empty; “data claims” must be grounded in actual outputs
- **What’s wrong:** `chapter4.tex` has placeholders; results claims are currently only in `instructions.md` and/or generated artifacts.
- **Correction:** In Chapter 4, include:
  - the exact dataset slice definitions used (same as the comparison scripts)
  - q-error summary (p50/p90/avg/max) for each model
  - training time and inference time
  - hardware/software environment (CPU, RAM, OS, Python version)
- **Ground truth recommendation:** Use the latest `compare_output/run_20260505_193206/` outputs as a baseline (or regenerate and cite the run ID used in the thesis).

### 8) Claim about XGBoost accuracy should mention heavy-tail failures
- **What’s wrong:** If the text claims “XGBoost is better” based only on p90 for training, it hides that the **average and max q-error can be much worse** due to extreme outliers.
- **Correction:** Report p50/p90 *and* average/max, and explain the failure mode (“ultra-fast queries” / near-zero targets causing extreme q-error sensitivity).

## Minor / hygiene

### 9) `main.tex` + Chapter 1 contain template placeholders and unrelated example code
- **What’s wrong:** Abstract text is placeholder; Chapter 1 includes a Keras/NLP code listing unrelated to the topic, plus generic template content.
- **Correction:** Remove template scaffolding and keep only domain-relevant figures, equations, and listings.

### 10) Some PDF text extraction issues are benign but should not affect citations
- **Observed:** A few PDFs extract with headers/page numbers as the first line (e.g., “72”, “222”), or garbled characters in the first line.
- **Correction:** No thesis change needed; just don’t use extracted-text artifacts as evidence for title/year/venue—prefer PDF metadata, the first page title block, or the official DOI/arXiv page.
