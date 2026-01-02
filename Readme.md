# LLM_Based_Chess_Agent_Analyzer

This repository contains a chess “move suggestion” agent powered by **LangGraph** + **Google Gemini (via LangChain)**, plus an **offline Stockfish-based evaluator** that generates **high-quality PNG report images** suitable for reports.

It was built to solve practical issues seen in LLM chess agents:
- **API rate limits (429)** and quota exhaustion
- **Illegal/invalid move outputs**
- Need for **repeatable evaluation** and **report-ready visuals**

The project provides:
- **Round-robin API key rotation** across multiple keys
- **Persistence of the last working key index** across runs (`.key_state.json`)
- **Multi-method test suites** (`direct`, `cot_hidden`, `sc_cot`)
- **Resumable runs** via per-run `state.json`
- **Per-node tracing** (`events.jsonl` + `run.log`)
- **Automatic repair node** to recover from invalid/illegal outputs
- **LangGraph diagram export** (`graph.png` + `graph.mmd`)
- **Stockfish evaluation** with centipawn-loss style scoring and labels
- **One PNG per test case** showing input / proposed / output boards + evaluation

---

## Table of Contents

- [1. Repository Layout](#1-repository-layout)
- [2. Installation](#2-installation)
- [3. API Keys and Rate Limit Handling](#3-api-keys-and-rate-limit-handling)
- [4. Running the Agent Suite](#4-running-the-agent-suite)
- [5. Testcases JSON Format](#5-testcases-json-format)
- [6. Agent Methods](#6-agent-methods)
- [7. LangGraph Workflow and Repair Logic](#7-langgraph-workflow-and-repair-logic)
- [8. Outputs: Results Directory Structure](#8-outputs-results-directory-structure)
- [9. Building Evaluation + Report Images (Stockfish)](#9-building-evaluation--report-images-stockfish)
- [10. Stockfish Scoring Explained](#10-stockfish-scoring-explained)
- [11. Logging and Debugging](#11-logging-and-debugging)
- [12. Git Hygiene (.gitignore Recommendations)](#12-git-hygiene-gitignore-recommendations)
- [13. Troubleshooting](#13-troubleshooting)

---

## 1. Repository Layout

Final files:

- `agent.py`  
  Runs a **suite** of chess test cases from a JSON file. For each case it executes multiple methods (`direct`, `cot_hidden`, `sc_cot`), stores results, logs, and saves LangGraph diagrams.

- `chess_core.py`  
  Core chess helpers built on **python-chess**:
  - `legal_moves_uci(fen)` → list of legal moves (UCI)
  - `validate_and_apply_move(fen, move_uci, method)` → validates + applies a move and returns a structured `AgentResult`.

- `evaluate_and_report.py`  
  Reads the outputs in `results/`, runs **Stockfish** evaluation for each method’s move, and generates **one image per test case**:
  - `reports/<suite_id>/<case_id>/comparison.png`

- `testcases.json`  
  The suite definition: suite-level defaults + a list of cases (FENs) and the methods to run.

- `requirements.txt`  
  Python dependencies.

Generated/runtime files (do not commit):

- `.env`  
  Contains your API keys (comma-separated). Must be ignored.

- `.key_state.json`  
  Persists the **global last good key index** across runs.

- `results/`  
  Raw run artifacts (state, logs, result JSON, diagrams, etc.)

- `reports/`  
  Final report images.

- `eval_debug.log`  
  Optional evaluation debug log output from `evaluate_and_report.py`.

---

## 2. Installation

### 2.1 Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
````

### 2.2 Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

* `python-dotenv>=1.0.0`
* `langchain-google-genai>=2.0.0`
* `langgraph>=0.2.0`
* `python-chess>=1.999`
* `Pillow>=10.0.0`
* `cairosvg>=2.7.0`

> Note: On some systems CairoSVG may require system Cairo libraries. If you get import/runtime errors around Cairo, install the OS packages for Cairo (varies by OS).

### 2.3 Install Stockfish (for evaluation)

Stockfish is required **only** if you run report generation with `--eval`.

Make sure `stockfish` is installed and available in your PATH, or pass `--stockfish-path /path/to/stockfish`.

---

## 3. API Keys and Rate Limit Handling

### 3.1 `.env` configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEYS=key1,key2,key3,key4
```

Rules:

* Keys are **comma-separated**
* Whitespace is trimmed
* You can use **many keys**; the rotation logic is designed for that

### 3.2 How key rotation works

The agent uses **Round Robin rotation**:

* Each LLM call uses the currently active key.
* If the call fails with **429 / RESOURCE_EXHAUSTED**, the agent **immediately switches** to the next key.
* It continues until a key works.
* After trying all keys once, that counts as **one full round**.

Controls:

* `retry_on_429` (bool): enable/disable key rotation behavior.
* `max_retries_per_key` (int): **number of full key rounds** allowed before cooling down.
* `cooloff_sec` (float): sleep time after exhausting rounds; then rounds restart.

### 3.3 Persisting the last good key index

The system persists the **last key that successfully worked**, so the next run starts from that key:

* `.key_state.json` (global, project root)
* `results/<suite_id>/suite_state.json` (per suite)

Example:

```json
{ "key_index": 4 }
```

This is updated on successful LLM calls.

---

## 4. Running the Agent Suite

The suite runner is `agent.py`.

### 4.1 Basic run

```bash
python agent.py --testcases testcases.json --results-dir results
```

This will:

* Load API keys from `.env`
* Load `testcases.json`
* For each case, run all requested methods
* Save per-run artifacts under `results/<suite_id>/...`

### 4.2 Resuming runs

Each method run stores its LangGraph state in:

* `results/<suite_id>/<case_id>/<method>/state.json`

If it exists, the agent loads it (resumable runs). This is especially useful for `sc_cot` sampling.

> Note: `summary.jsonl` is append-only; repeated runs append additional records. If you want a clean run history, delete the suite directory under `results/` before re-running.

---

## 5. Testcases JSON Format

A testcase file contains:

* `suite_id`: output folder name under `results/`
* `defaults`: parameters applied to all cases/methods (unless overridden)
* `cases`: list of test cases

Each case must include:

* `case_id` (string, unique)
* `fen` (string)
* `method` (string) or `methods` (array)

Optional:

* `per_method`: per-method overrides for that case
* Case-level overrides of any defaults

Example per-method override (common for `sc_cot`):

```json
"per_method": {
  "sc_cot": { "min_interval_sec": 3.0, "sc_samples": 7 }
}
```

### 5.1 Parameter reference

These parameters may appear in `defaults`, a case object, or under `per_method[method]`:

* `model` (string)
  Gemini model name (example: `"gemini-2.5-flash-lite"`)

* `temperature` (float)
  Sampling temperature.

* `min_interval_sec` (float)
  Rate limiter delay between LLM calls (per run).

* `retry_on_429` (bool)
  Enable key rotation + cooloff logic.

* `max_retries_per_key` (int)
  Number of full key rounds allowed before cooloff.

* `cooloff_sec` (float)
  Sleep time after exhausting all rounds.

* `max_attempts` (int)
  For `direct` and `cot_hidden`: attempts to obtain a legal move.

* `sc_samples` (int)
  For `sc_cot`: number of samples to draw.

* `max_repairs` (int)
  Maximum number of repair attempts allowed.

---

## 6. Agent Methods

The suite can run multiple strategies for the same position.

### 6.1 `direct`

* Prompts the model to choose a move from `LEGAL_MOVES_UCI`.
* No explicit “internal reasoning” instruction.

### 6.2 `cot_hidden`

* Asks the model to “analyze internally step-by-step” but **must not reveal reasoning**.
* Output must be JSON only:

  ```json
  {"move_uci":"..."}
  ```

### 6.3 `sc_cot` (self-consistency voting)

* Samples the model `sc_samples` times.
* Each sample must output valid JSON and a move in `LEGAL_MOVES_UCI`.
* Valid moves become votes.
* Final move is chosen by majority vote.

Artifacts saved for `sc_cot` include:

* `sc_raw`: raw model outputs for each sample
* `sc_votes`: filtered valid moves
* `vote_distribution`: counts per move

If **no valid votes** are produced:

* The method fails with `"No valid samples produced."`
* Repair logic can attempt recovery (up to `max_repairs` times).

---

## 7. LangGraph Workflow and Repair Logic

### 7.1 Nodes

The agent uses a LangGraph `StateGraph` with nodes:

1. `get_legal_moves`
   Computes legal moves from the input FEN via `python-chess`.

2. `init_sc`
   Initializes SC fields (if needed) and `repair_attempts`.

3. `sc_step` (only for `sc_cot`)
   Executes one sample, appends raw output to `sc_raw`, and if valid adds to `sc_votes`.

4. `choose_move`

   * For `sc_cot`: chooses the winning vote (or sets an error if none).
   * For `direct`/`cot_hidden`: retries inside this node up to `max_attempts`.

5. `validate_apply`
   Validates and applies move using `validate_and_apply_move()` and writes `result.json`.

6. `repair_move` (conditional)
   On invalid/illegal output, generates a stricter feedback prompt and tries to recover a legal move.

### 7.2 Conditional routing

* After `init_sc`:

  * If `method == "sc_cot"` → `sc_step`
  * Else → `choose_move`

* `sc_step` loops until `sc_i == sc_samples`

* After `validate_apply`:

  * If valid → END
  * Else if repairable and `repair_attempts < max_repairs` → `repair_move`
  * Else → END

### 7.3 What is considered “repairable”

Errors matching any of these markers are treated as repairable:

* `"no valid samples produced"`
* `"did not return a valid move"`
* `"illegal"`
* `"not valid"`
* `"invalid"`

### 7.4 Diagram export

On each method run, the agent attempts to save the graph diagram:

* Suite-level:

  * `results/<suite_id>/graph.png`
  * `results/<suite_id>/graph.mmd`

* Per run:

  * `results/<suite_id>/<case_id>/<method>/graph.png`
  * `results/<suite_id>/<case_id>/<method>/graph.mmd`

If diagram rendering fails, it logs a warning and continues.

---

## 8. Outputs: Results Directory Structure

Example structure:

```
results/
  suite_combo_001/
    graph.png
    graph.mmd
    suite_state.json
    summary.json
    summary.jsonl

    medium_middlegame_tactical/
      methods_index.json

      direct/
        state.json
        events.jsonl
        result.json
        run.log
        graph.png
        graph.mmd

      cot_hidden/
        ...

      sc_cot/
        ...
```

### 8.1 `result.json` format

On success:

```json
{
  "input_fen": "...",
  "move_uci": "e2e4",
  "move_san": "e4",
  "fen_after": "...",
  "method": "direct",
  "valid": true,
  "error": null,
  "meta": { "key_index": 4 },
  "sc_info": null
}
```

For `sc_cot`, `sc_info` is populated:

```json
"sc_info": {
  "samples_requested": 7,
  "samples_taken": 7,
  "valid_samples": 6,
  "vote_distribution": { "g1f3": 6 }
}
```

### 8.2 `state.json`

Stores the full agent state, including:

* `legal_moves`
* `sc_votes`, `sc_raw`, `sc_i` (for `sc_cot`)
* `repair_attempts`
* `last_prompt_feedback`

### 8.3 `events.jsonl`

Append-only per-node trace events. Useful for tracing the execution path and SC progress.

### 8.4 `methods_index.json`

Per case file listing each method output directory + success/error metadata, used by the report builder.

---

## 9. Building Evaluation + Report Images (Stockfish)

The report builder is `evaluate_and_report.py`.

It creates one PNG per case:

```
reports/<suite_id>/<case_id>/comparison.png
```

### 9.1 With evaluation enabled (recommended)

```bash
python evaluate_and_report.py \
  --results-dir results \
  --out-dir reports \
  --suite-id suite_combo_001 \
  --coordinates \
  --eval \
  --stockfish-depth 20 \
  --log-level INFO \
  --log-file eval_debug.log
```

### 9.2 Without evaluation (boards + metadata only)

```bash
python evaluate_and_report.py \
  --results-dir results \
  --out-dir reports \
  --suite-id suite_combo_001 \
  --coordinates
```

### 9.3 Important options

* `--board-size` (default `520`)
  Larger boards produce clearer report images.

* `--dpi` (default `300`)
  Ideal for printing and Word reports.

* `--coordinates`
  Adds board coordinates.

* `--stockfish-path`
  Specify Stockfish binary if not in PATH.

* `--stockfish-depth`
  Depth controls scoring accuracy vs runtime.

* `--stockfish-threads`, `--stockfish-hash-mb`
  Engine performance settings.

---

## 10. Stockfish Scoring Explained

Stockfish reports `score cp` from the **side-to-move** perspective (STM).

To measure how good the agent’s move was, the report computes:

1. Evaluate the position **before** the move:

   * `before_cp_mover = cp_before_stm`

2. Apply the agent move and evaluate **after**:

   * Stockfish returns `cp_after_stm` for the opponent (now side-to-move)
   * Convert to mover perspective:

     * `after_cp_mover = -cp_after_stm`

3. Compute “drop” (centipawn loss style):

   * `drop = max(0, before_cp_mover - after_cp_mover)`

Classification thresholds:

* `drop < 15` → **Excellent**
* `< 50` → **Good**
* `< 120` → **Inaccuracy**
* `< 250` → **Mistake**
* `>= 250` → **Blunder**

The report also displays:

* Best move by Stockfish (UCI + SAN)
* Depth used
* Before/after values in mover perspective

---

## 11. Logging and Debugging

### 11.1 Agent logs

Per method run:

* `results/<suite_id>/<case_id>/<method>/run.log`

Includes:

* Node execution sequence
* SC progress
* Key rotation warnings and active key index
* Repair attempt logs
* Diagram generation status

### 11.2 Evaluation logs

Optional evaluation debug output:

* `--log-level INFO|DEBUG`
* `--log-file eval_debug.log`

Includes:

* Start/end of each evaluation
* Drop, label, best move
* Before/after mover-perspective scores
* FEN debug lines at DEBUG level

---

## 12. Git Hygiene (.gitignore Recommendations)

Recommended `.gitignore` entries:

```gitignore
# Secrets
.env

# Runtime key state
.key_state.json

# Outputs
results/
reports/

# Logs
*.log

# Python
__pycache__/
*.pyc
.venv/
```

---

## 13. Troubleshooting

### 13.1 Repeated 429 / RESOURCE_EXHAUSTED

This is expected when a key hits quota.

Mitigations:

* Add more keys to `GOOGLE_API_KEYS`
* Increase `cooloff_sec`
* Increase `min_interval_sec`
* Reduce `temperature` or decrease total sampling (especially `sc_samples`)

### 13.2 `sc_cot` produced no valid samples

If `sc_votes` is empty, you’ll get `"No valid samples produced."`.

The repair node can recover if:

* `max_repairs > 0`
* Error is considered repairable

### 13.3 Stockfish not found

If Stockfish is not in PATH:

```bash
python evaluate_and_report.py --eval --stockfish-path /full/path/to/stockfish
```

### 13.4 CairoSVG issues

If `cairosvg` fails:

* Ensure it is installed: `pip install cairosvg`
* Install OS-level Cairo dependencies if required by your system

### 13.5 LangGraph diagram render fails

If diagram rendering fails, the agent logs:

* `Could not save LangGraph diagram: ...`

The run still completes. If you need diagrams:

* Ensure your environment supports the Mermaid rendering method available in your LangGraph version.
