# Plan: Architecture-A End-to-End Test Script

## Goal
Create `scripts/test_architecture_a.py` that tests the full pipeline: KalshiBench questions → Exa evidence → 3 LLM agents → majority vote → comparison with ground truth.

## Output Format
CSV file with separate columns for each field:

| Column | Description |
|--------|-------------|
| `question_id` | Unique question identifier |
| `question_text` | The prediction market question |
| `resolution_criteria` | How the question resolves |
| `category` | Question category (Politics, Sports, etc.) |
| `ground_truth` | Actual resolution (YES/NO) |
| **GPT-4o Agent** | |
| `gpt4o_decision` | GPT-4o's decision (YES/NO) |
| `gpt4o_confidence` | GPT-4o's confidence (0.0-1.0) |
| `gpt4o_reasoning` | GPT-4o's full reasoning text |
| `gpt4o_error` | Error message if agent failed |
| **Claude Agent** | |
| `claude_decision` | Claude's decision (YES/NO) |
| `claude_confidence` | Claude's confidence (0.0-1.0) |
| `claude_reasoning` | Claude's full reasoning text |
| `claude_error` | Error message if agent failed |
| **Gemini Agent** | |
| `gemini_decision` | Gemini's decision (YES/NO) |
| `gemini_confidence` | Gemini's confidence (0.0-1.0) |
| `gemini_reasoning` | Gemini's full reasoning text |
| `gemini_error` | Error message if agent failed |
| **Aggregated Results** | |
| `final_decision` | Majority vote result (YES/NO) |
| `yes_votes` | Number of YES votes |
| `no_votes` | Number of NO votes |
| `is_correct` | True if final_decision == ground_truth |

Optional JSON output includes summary statistics.

## CLI Arguments
```bash
python scripts/test_architecture_a.py \
  -n 10                          # Number of questions (default: 10)
  --question-ids ID1 ID2         # Specific question IDs (overrides -n)
  --category Politics            # Filter by category
  --cache-only                   # Only use cached evidence
  --cache-dir cache/evidence     # Evidence cache directory
  -o results/output.csv          # Output CSV path
  --json results/output.json     # Optional JSON with summary stats
  --dry-run                      # Show what would run without executing
  -v                             # Verbose progress output
```

## Implementation Steps

### 1. Create data classes for results
- `QuestionResult`: Per-question results with all agent decisions and evaluation
- `RunSummary`: Aggregate statistics (accuracy, agreement rates, timing, cost)

### 2. Core functions
- `get_evidence(question, retriever, cache_only)`: Load from cache or retrieve via Exa
- `process_question(question, evidence, runner)`: Run MultiAgentRunner, extract per-agent results
- `compute_summary(results, runtime)`: Calculate accuracy, agreement, timing metrics

### 3. Output functions
- `write_csv(results, path)`: Write detailed results
- `write_json(results, summary, path)`: Write results + summary statistics
- `print_summary(summary)`: Console output with formatted stats

### 4. Main function
- Parse CLI args
- Load/filter KalshiBench questions
- Show cost estimate warning
- Process each question with progress bar (tqdm)
- Write outputs and print summary

## Key Files to Use
| File | Purpose |
|------|---------|
| `src/data/kalshi_loader.py` | `load_kalshi_bench()`, `KalshiQuestion` |
| `src/retrieval/exa_retriever.py` | `ExaOracleRetriever.load_from_cache()`, `retrieve_and_cache()` |
| `src/resolution/runner.py` | `MultiAgentRunner.resolve()` |
| `src/agents/base.py` | `AgentResolution`, `Decision` enum |

## Important Implementation Notes

### 1. Agent ordering (completion order vs fixed order)
`MultiAgentRunner.resolve()` uses `as_completed(futures)`, so `AggregatedResolution.agent_resolutions` is in **completion order**, not fixed GPT → Claude → Gemini order. When populating CSV columns, **match by `agent_name`** (e.g., `"gpt-4o"`, `"claude-haiku"`, `"gemini-2.0-flash"`), not by list index:
```python
agent_results = {r.agent_name: r for r in aggregated.agent_resolutions}
gpt4o = agent_results.get("gpt-4o")
claude = agent_results.get("claude-haiku")
gemini = agent_results.get("gemini-2.0-flash")
```

### 2. Ground truth normalization
`KalshiQuestion.ground_truth` is lowercase (`"yes"` / `"no"`) from the dataset, while `Decision.YES.value` is uppercase. **Normalize before comparison**:
```python
ground_truth_normalized = question.ground_truth.strip().upper()
is_correct = (final_decision == ground_truth_normalized)
```

### 3. CSV escaping
`question_text`, `resolution_criteria`, and `*_reasoning` fields can contain commas, newlines, and quotes. Use `csv.DictWriter` with proper quoting:
```python
writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
```

### 4. Loading questions by ID
There's no `load_by_ids()` function. When `--question-ids` is used, load the full dataset and filter:
```python
all_questions = load_kalshi_bench()
question_map = {q.id: q for q in all_questions}
questions = [question_map[qid] for qid in args.question_ids if qid in question_map]
```

## Error Handling
- Evidence retrieval failures: Skip question, log warning
- Agent API failures: Already handled by BaseAgent (returns AgentResolution with error field)
- Partial failures: Majority vote works with 2/3 agents succeeding

## Summary Statistics (printed + JSON)
- Total/successful/failed questions
- Overall accuracy (majority vote)
- Per-agent accuracy (GPT-4o, Claude, Gemini)
- Agreement rates (unanimous 3/3, majority 2/3)
- Average latency, total runtime
- Estimated API cost

## Verification
```bash
# Test with 3 questions using cached evidence
python scripts/test_architecture_a.py -n 3 --cache-only -v

# Check output file exists and has correct columns
head -1 results/architecture_a_results.csv

# Run with JSON to verify summary stats
python scripts/test_architecture_a.py -n 5 --cache-only --json results/test.json
cat results/test.json | python -m json.tool | head -30
```

## Dependencies
- `tqdm` for progress bar (already in project based on cache_all_evidence.py pattern)
