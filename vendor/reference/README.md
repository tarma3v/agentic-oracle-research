# Multi-Agent AI Oracle System for Prediction Market Resolution

Paper codebase for evaluating multi-agent LLM systems on resolved Kalshi prediction markets. The repository compares two main resolution pipelines built on a shared evidence retrieval layer so model reasoning can be studied separately from retrieval quality.

## What This Repo Contains

- **Architecture A**: independent agent judgments followed by aggregation
- **Architecture B**: two-round debate with peer reasoning and final aggregation
- **Shared retrieval**: Exa-backed evidence collection with cache support
- **Analysis tooling**: scripts for aggregate evaluation, market-instance analysis, and error analysis
- **Paper artifacts**: methodology notes, figures, notebooks, and generated results

## System Overview

```text
KalshiBench question
  -> Exa retrieval / cached evidence
  -> 3 LLM agents with shared evidence
  -> aggregation or debate-based resolution
  -> CSV / JSON outputs
  -> downstream analysis scripts
```

The default agent trio in the current codebase is:

- `gpt-5-nano`
- `deepseek-v3`
- `llama-3.3-70b-turbo`

Optional runs can swap in Claude and/or Gemini through the experiment scripts.

## Setup

### 1. Install dependencies

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -e ".[dev]"
```

Alternatively, [setup_venv.sh](setup_venv.sh) creates a `venv/` directory and installs the project plus test dependencies.

### 2. Create `.env`

Copy the committed template, then add only the keys for services you plan to use:

```bash
cp .env.example .env
```

The default three-model pipeline requires:

- `OPENAI_API_KEY`
- `TOGETHER_API_KEY`
- `EXA_API_KEY`

`ANTHROPIC_API_KEY` is needed only with `--use-claude`; `GOOGLE_API_KEY` is needed only with `--use-gemini`.

### 3. Evaluation dataset

The standard evaluation input is committed at `data/kalshibench_v2_evaluation_Filtered.csv` (1,189 rows) and is the default input for both evaluation scripts. No Hugging Face download is needed for the standard workflow.

To download a local copy of the complete upstream KalshiBench v2 dataset for other analyses, run:

```bash
python scripts/download_kalshibench_v2.py
```

This stores the dataset in `cache/kalshibench-v2/`. The loader will prefer that local copy when present.

### 4. Optional evidence prefetch

```bash
python scripts/cache_all_evidence.py
```

This requires `EXA_API_KEY` and populates `cache/evidence/` for all 1,189 evaluation rows. Later runs can use `--cache-only` to avoid Exa retrieval requests; model-provider keys are still required.

## Quick Start

Check the default configuration without making API calls:

```bash
python scripts/test_architecture_a.py --dry-run
python scripts/test_architecture_b.py --dry-run
```

Run a five-question live sample. This calls Exa and the three model providers, so it requires the default API keys and incurs provider costs:

```bash
python scripts/test_architecture_a.py -n 5 -o results/reproduced/architecture_a_sample.csv
python scripts/test_architecture_b.py -n 5 -o results/reproduced/architecture_b_sample.csv
```

After evidence prefetching, run a small cached sample:

```bash
python scripts/test_architecture_a.py -n 5 --cache-only -o results/reproduced/architecture_a_cached_sample.csv
python scripts/test_architecture_b.py -n 5 --cache-only -o results/reproduced/architecture_b_cached_sample.csv
```

Run the full evaluation set with explicit output paths:

```bash
python scripts/test_architecture_a.py -n 1189 -o results/reproduced/architecture_a_full.csv
python scripts/test_architecture_b.py -n 1189 -o results/reproduced/architecture_b_full.csv
```

Useful flags for both experiment scripts:

- `--retrieval-mode highlights|full_text`
- `--question-ids ...`
- `--category ...`
- `--start-row ...`
- `--cache-start-row ...`
- `--use-claude`
- `--use-gemini`

See [prompt templates and the complete CLI reference](docs/prompt_templates_and_cli.md) for the paper-facing prompt text and every supported evaluation flag.

## Analysis Entry Points

Comprehensive A/B analysis:

```bash
python scripts/analyze_architecture_results.py \
  --arch-a results/Final_Architecture_A_results.csv \
  --arch-b results/Final_Architecture_B_results.csv \
  --output-dir results/comprehensive_analysis
```

Market-instance analysis for a results file:

```bash
python scripts/analysis_by_market_instance.py \
  --input-csv results/Final_Architecture_B_results.csv \
  --market-key question_id \
  --output-dir results/market_instance_analysis_final_b
```

Error-analysis utilities live in [scripts/error_analysis/README.md](scripts/error_analysis/README.md). These scripts export focused slices such as hallucination candidates, retrieval failures, shared-bias cases, temporal conflicts, and failure-case CSVs.

## Repo Map

```text
src/                     Core package code
  agents/                LLM wrappers and structured-output adapters
  retrieval/             Exa retrieval and evidence packet handling
  resolution/            Aggregation and debate runners
  data/                  KalshiBench loading
  evaluation/            Evaluation helpers
  utils/                 Shared utilities

scripts/                 Experiment and analysis entry points
scripts/error_analysis/  Focused error-export scripts
tests/                   Automated tests

results/                 Generated experiment outputs and analysis artifacts
data/                    Committed 1,189-row evaluation dataset
figures/                 Curated paper figures
notebooks/               Notebook-based analysis and figure generation
docs/                    Paper notes and supporting materials
cache/                   Local dataset and retrieval caches
```

## Notes

- This repository intentionally keeps many generated research artifacts under `results/` for reproducibility and paper reference.
- Some historical filenames in `results/` reflect earlier experiment stages; the README examples above use explicit paths to the current main artifacts.

## License

MIT
