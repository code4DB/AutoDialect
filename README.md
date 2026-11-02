# AutoDialect

## Overview

AutoDialect is an LLM-based system for synthesizing and validating reusable
SQL dialect translation rule implementations from source-target SQL pairs.
For each case, it aligns the source SQL, the current translation, and the
expected target SQL; extracts context-complete SQL units; and groups related
units into budget-constrained generation tasks. Parser checks, focused-unit
checks, and complete-query checks act as validation gates for generation and
repair, with repair restricted to the detected failure scope.

The command-line pipeline applies generated rule changes to a supplied
translation-engine checkout. It records the changes, validation evidence, and
stage trajectories needed to inspect each case. `rule_main.py` performs rule
generation and validation, while `rule_eval.py` rebuilds clean worktrees and
replays the recorded changes for evaluation.

## Repository Layout

```text
autodialect/
├── rule_main.py                 # Rule generation entrypoint
├── rule_eval.py                 # Evaluation entrypoint
├── agent_runners/               # Agent execution wrapper
├── code_utils/                  # Generation, validation, and memory utilities
│   ├── memory/
│   └── sql_ut/
├── prompt/                      # Generation prompts
├── autodialect_agent/           # Bundled agent runtime
├── autodialect_config.yaml      # Default agent configuration
├── LICENSES/                    # License texts
├── .env.example                 # Environment variable template
└── requirements.txt
```

## Environment Setup

### Requirements

- Python 3.12
- Git available on `PATH`
- A Unix-like shell environment
- A local checkout of the supported rule-based translation engine
- An OpenAI-compatible chat-completions endpoint with tool-calling support

### Installation

Run the following commands from the `autodialect/` directory:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Model Configuration

Copy the environment template and replace its placeholder values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
| --- | --- | --- |
| `AUTODIALECT_API_KEY` | Yes | API credential |
| `AUTODIALECT_MODEL` | Yes | Model identifier |
| `AUTODIALECT_BASE_URL` | Yes | OpenAI-compatible API base URL |
| `AUTODIALECT_TESTCASE_FILE` | No | Default online-case JSON path |
| `AUTODIALECT_ENGINE_SOURCE` | No | Default translation-engine checkout path |
| `AUTODIALECT_PYTHON_BIN` | No | Python executable used by subprocesses |
| `AUTODIALECT_STAGE_TIMEOUT_S` | No | Positive per-stage timeout in seconds |

The default agent settings are stored in `autodialect_config.yaml`. Shell
variables take precedence over values in `.env` and `.env.local`.

### Optional Memory Retrieval

Runtime memory is disabled by default. To enable read-only memory retrieval,
install `letta-client`, set `enable_memory: true` under
`agents.autodialect_agent` in `autodialect_config.yaml`, and configure:

- `LETTA_BASE_URL`;
- either `LETTA_AGENT_ID` or `LETTA_AGENT_NAME`; and
- `LETTA_API_KEY` when required by the service.

## Usage

### Input Format

The input is a non-empty JSON list in which each object represents one online
translation case:

```json
[
  {
    "id": "example_001",
    "source_dialect": "mysql",
    "target_dialect": "postgres",
    "source_sql": "SELECT IFNULL(amount, 0) FROM orders",
    "expected_sql": "SELECT COALESCE(amount, 0) FROM orders"
  }
]
```

| Field | Type | Description |
| --- | --- | --- |
| `id` | string or integer | Unique case identifier |
| `source_dialect` | string | Source dialect name |
| `target_dialect` | string | Target dialect name |
| `source_sql` | string | Source SQL statement |
| `expected_sql` | string | Expected target SQL statement |

Case identifiers must be unique, the source and target dialects must differ,
and both SQL fields must be non-empty. Dialect names must be recognized by the
supplied translation-engine version.

### Rule Generation

The following example reads online cases, uses `worktrees/online` as the
per-case worktree prefix, and writes generation records under
`results/online/`:

```bash
python rule_main.py \
  --testcase-file /path/to/online_cases.json \
  --src-dir /path/to/translation-engine \
  --dst-dir ./worktrees/online \
  --result-folder ./results/online \
  --result-file ./results/online/results.json \
  --autodialect-config-file ./autodialect_config.yaml
```

Add `--limit 1` for a one-case smoke run. Run
`python rule_main.py --help` for the complete command-line interface.

For each case, AutoDialect validates the input and engine checkout, checks
source and target parsing, constructs focused SQL-unit groups, generates rule
changes, and validates both the focused units and complete statements before
persisting the result. Comparisons normalize letter case, whitespace, and a
trailing semicolon; other SQL text differences remain significant.

### Evaluation

Evaluation replays generated changes in clean worktrees:

```bash
python rule_eval.py \
  --testcase-file /path/to/online_cases.json \
  --src-dir /path/to/translation-engine \
  --dst-dir ./worktrees/online \
  --result-folder ./results/online \
  --result-file ./results/online/results.json \
  --validate-splits
```

The principal validation controls are:

- `--validate-splits` to validate focused split and group units;
- `--validate-variants` to also validate generated SQL variants;
- `--limit N` to evaluate the first `N` cases; and
- `--case-indices-file PATH` to evaluate listed zero-based case indices.

Run `python rule_eval.py --help` for all options.

### Outputs

With the paths used above, AutoDialect produces:

| Path | Content |
| --- | --- |
| `worktrees/online_<source>_<target>_<index>/` | Per-case generation worktrees |
| `worktrees/online/` | Clean worktree rebuilt and reused during evaluation |
| `results/online/results.json` | Generation and validation records |
| `results/online/<source>_<target>_<index>/` | Generated translation-engine files |
| `results/online/trajectories/` | Per-stage trajectories and manifests |
| `results/online/results_validated.json` | Evaluation output |

Each result record contains the input case, generated file changes, execution
status, validation evidence, and trajectory references.

## Verification and Reproducibility

Run the following checks from the `autodialect/` directory:

```bash
python -m compileall -q .
python rule_main.py --help
python rule_eval.py --help
```

For a reproducible generation and evaluation run, record the Python and
dependency versions, translation-engine revision, model identifier, input
file checksum, command-line options, and optional runtime settings. Reuse the
same case file, case order, output paths, and validation flags when comparing
runs.
