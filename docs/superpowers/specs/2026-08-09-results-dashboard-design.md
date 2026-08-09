# Design: Results Dashboard

**Date:** 2026-08-09
**Status:** Draft — pending review

## Problem

As milestones 4-9 land, `results/raw/` accumulates CSVs with different shapes
(load-sweep throughput data now, LongBench perplexity/accuracy later,
eventually a second model/hardware pair) and the real story of the project
— design decisions, hardware constraints discovered, bugs hit and fixed —
lives scattered across commit messages and two markdown docs. There's no
single place to look at all of it.

## Scope

A local-only, single-user record of everything: no accounts, no server.

`results/dashboard.py` scans the repo and renders one self-contained,
offline-viewable HTML file, `results/dashboard.html`. Re-run the script any
time new data lands. The output is derived/regenerable, so it's gitignored
like other build artifacts — the script itself is the tracked deliverable.

## Content

1. **Trials** — one card per file in `results/raw/*.csv`. If the columns
   match the loadgen sweep schema (`loadgen/runner.py:CSV_FIELDS`), render
   summary stats (success rate, p50/p95 latency, mean TTFT) plus a small
   inline-SVG latency-distribution chart. Anything else (future eval CSVs
   with a different schema) falls back to a plain sortable table, so the
   dashboard doesn't break when milestone 6 adds a new CSV shape.
2. **Milestone status** — the 9 milestones from the design doc, each marked
   done/pending by checking for the commit/file that milestone's plan task
   produces (e.g. milestone 4 done ⇔ `serving/server.py` exists and its
   commit is in the log).
3. **Narrative** — a timeline built from `git log` commit subjects and
   bodies (already written to explain real issues hit and fixed, e.g. the
   flashinfer/CUDA incompatibility), plus links to the design doc and plan
   under `docs/superpowers/`.

## Non-goals

- No authentication, no multi-user access, no hosting/deployment.
- No live server — regenerate-on-demand only.
- No new runtime dependencies beyond `pandas` (already required).

## Implementation notes

- `pandas.read_csv` for parsing; schema detection by comparing column sets
  against known schemas (starting with just the loadgen one).
- Charts as hand-written inline SVG (no matplotlib-to-PNG, no JS charting
  library) — keeps the output small, crisp, and dependency-free so it opens
  via plain `file://` with no internet access needed, matching the
  project's reproducibility-first stance.
- Visual design continues the same systems/technical-report look (slate
  blue + amber accent, serif headings, monospace data) used in the design
  doc's own rendered artifact, for visual continuity across the project.
