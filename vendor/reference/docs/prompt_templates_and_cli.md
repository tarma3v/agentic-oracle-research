# Prompt Templates and Command-Line Reference

This document preserves the prompt templates and command-line interface described in the paper appendix. It is a paper-facing reference. The executable prompt definitions and argument parsers remain the implementation source of truth:

- `src/agents/prompts.py` for Architecture A
- `src/resolution/debate/prompts.py` for Architecture B
- `scripts/test_architecture_a.py` and `scripts/test_architecture_b.py` for CLI behavior

## Architecture A Prompt Template

### System prompt

```text
You are an expert prediction market resolution agent. Your task is to determine whether a prediction market question should resolve to YES or NO based on the provided evidence.

Instructions:
1. Read the question and resolution criteria carefully.
2. Analyze ALL provided sources for relevant information.
3. Make your decision based on definitive evidence. Prioritize information describing outcomes that have already occurred.
4. If evidence is ambiguous, use your best judgment.
5. Rate your confidence from 0.0 (very uncertain) to 1.0 (absolutely certain).

Output Format:
- decision: YES or NO
- confidence: 0.0 to 1.0
- reasoning: Explanation referencing specific sources
```

### User-message template

```text
Please analyze the following prediction market question and evidence, then provide your resolution decision.

{evidence_text}

Based on the evidence above, should this question resolve to YES or NO?
```

## Architecture B Prompt Templates

### Round 1 system prompt

```text
You are an expert prediction market resolution agent. Your task is to determine whether a prediction market question should resolve to YES or NO based on the provided evidence.

Determine whether the question should resolve to YES or NO based only on the evidence provided.
```

### Round 1 user-message template

```text
QUESTION: {question}

RESOLUTION CRITERIA: {criteria}

EVIDENCE PROVIDED from reputable sources:
{exa_evidence}

Provide your answer in JSON with these fields:
- decision: YES or NO
- confidence: number between 0.0 and 1.0
- reasoning: detailed step-by-step reasoning that cites relevant evidence and criteria interpretation

Think carefully about:
1. Read the question and resolution criteria carefully
2. Analyze ALL provided sources for relevant information
3. Make your decision based on definitive evidence. You should prioritize information that describes an outcome that happened.
4. If evidence is ambiguous, use your best judgment
5. Rate your confidence in your decision from 0.0 (very uncertain) to 1.0 (absolutely certain)
```

### Round 2 system prompt

```text
You are in the final cross-examination round of a multi-agent resolution debate.
You will see how other agents reasoned about the same question. Your job is to CRITICALLY EVALUATE their reasoning against the evidence -- not to defer to them.

IMPORTANT: You should only change your answer if you identify a SPECIFIC factual error in your own round 1 reasoning, or if another agent points to SPECIFIC EVIDENCE in the shared packet that you misread or overlooked. Do NOT change your answer simply because other agents disagree with you or sound confident.
```

### Round 2 user-message template

```text
You previously resolved this prediction market question.
Now you will see how two other expert agents reasoned about the same question.

QUESTION: {question}

RESOLUTION CRITERIA: {criteria}

EVIDENCE (SAME SHARED PACKET FROM ROUND 1):
{exa_evidence}

YOUR PREVIOUS ANSWER:
{your_round_1_response}

OTHER AGENTS' REASONING:
{other_agents_section}

This is the FINAL round. Critically evaluate the other agents' reasoning:

1. For each agent that DISAGREES with you: Do they cite specific evidence from the shared packet that contradicts your reasoning? Or are they asserting a conclusion without evidentiary support?
2. Re-read the specific pieces of evidence that are most relevant to the disagreement.
3. ONLY change your decision if you can identify a concrete error in your own round 1 analysis, for example, you misread a date, overlooked a source, or misinterpreted the resolution criteria.
4. If agents agree with you, do NOT increase your confidence unless they provide additional evidence-based reasoning you hadn't considered.

DEFAULT BEHAVIOR: Change your decision ONLY if another agent identifies specific evidence
or because there is a concrete flaw in your reasoning, not simply because they reached
a different conclusion.

Provide your FINAL answer in JSON with these fields:
- decision: YES or NO
- confidence: number between 0.0 and 1.0
- reasoning: explain your final reasoning, explicitly addressing each disagreeing agent's key claim
- revised: true if you changed your decision from round 1, else false
- convergence_notes: short note on agreement/disagreement and why
```

## Command-Line Interface

Both evaluation scripts default to `data/kalshibench_v2_evaluation_Filtered.csv`, the committed 1,189-row evaluation dataset. See the root README for runnable examples.

| Flag | Applies to | Default | Description |
| --- | --- | --- | --- |
| `-n` | A, B | `10` | Number of questions to evaluate. Use `-n 1189` for the full committed evaluation dataset. |
| `--questions-csv` | A, B | `data/kalshibench_v2_evaluation_Filtered.csv` | Path to the input questions CSV. |
| `--start-row` | A, B | `1` | One-based row number at which to begin evaluation. |
| `--cache-start-row` | A, B | none | One-based row to start cache-key lookup without skipping input rows. |
| `--question-ids` | A, B | none | Specific question IDs to evaluate; overrides `-n`. |
| `--category` | A, B | none | Restrict the input to one category, such as `Politics` or `Sports`. |
| `--retrieval-mode` | A, B | `highlights` | Exa retrieval mode: `full_text` or `highlights`. |
| `--num-results` | A, B | `10` | Number of Exa sources retrieved for each question. |
| `--fulltext-max-characters` | A, B | `4000` | Per-source character cap in `full_text` mode. |
| `--highlights-max-characters` | A, B | `2000` | Character cap for Exa highlight snippets; pass `0` to use Exa defaults. |
| `--cache-only` | A, B | false | Use cached evidence only; cache misses are skipped. |
| `--cache-dir` | A, B | `cache/evidence` | Directory containing cached evidence packets. |
| `--use-claude` | A, B | false | Replace the DeepSeek agent slot with Claude. |
| `--use-gemini` | A, B | false | Replace the Llama agent slot with Gemini. |
| `-o`, `--output` | A | timestamped path | Output CSV path. |
| `-o`, `--output` | B | `results/architecture_b_results.csv` | Output CSV path. |
| `--json` | A, B | none | Optional output path for aggregate summary JSON. |
| `--dry-run` | A, B | false | Print the planned configuration without making API calls. |
| `-v`, `--verbose` | A, B | false | Enable per-question progress logging. |
