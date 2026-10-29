# Agent Gym + Replay

[![CI](https://github.com/jingwenliu0113-afk/agent-gym/actions/workflows/ci.yml/badge.svg)](https://github.com/jingwenliu0113-afk/agent-gym/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)

**English** | [繁體中文](README.zh-TW.md)

**When an LLM agent fails a multi-step task, at which step did it become unrecoverable, and did it do anything unsafe
on the way, even when it succeeded?**

```text
Task → Agent actions → state changes → grader → replay from checkpoint
```

A final pass / fail score cannot answer either question. This project builds a deterministic, resettable environment
in which every step of an agent's run can be restored and replayed, and uses it to estimate
**P(success | checkpoint k)**: the chance that the agent still succeeds if it is restarted from step *k* of a failed
run. A sudden drop in that curve locates the step where the run became unrecoverable.

## At a glance

- **Environment.** A deterministic Issue Tracker (issues, users, comments, labels; 5 tools) with a canonical state
  hash and snapshot / restore at any step. No real systems are touched.
- **Recorder and replay.** Every step records the model message, tool call, result and the state hash before and
  after. Restoring a checkpoint rebuilds both the environment and the conversation, byte for byte, and the agent
  rolls out again from there.
- **Grader.** Final-state correctness is scored separately from process safety (forbidden actions and attempts,
  constraint violations, unnecessary mutations, tool errors), so a run can succeed and still be unsafe.
- **Failure localization.** A coarse-to-fine checkpoint sweep estimates P(success | k) with Wilson intervals and
  classifies each curve as a *cliff* (a sudden, significant drop), *gradual* or *undecidable*, using a criterion
  chosen before any LLM result (`bonferroni95`, plan (2, 10, 48, 0.3); decision 025).
- **Agents.** The core agent is a frozen, self-hosted model (`qwen3.5:9b-q4_K_M` on Ollama, on a remote GPU used under
  a shared lock). Claude Haiku 5.5 (Anthropic API) was added as an optional, budget-capped second agent.
- **Headline results** (each study pre-registered and frozen before its evaluation set was run once):
  - **Round 1** (15 tasks): **45/45** episodes succeeded, all safe; **480 reruns** found no cliff. The Core Gate result
    is `gradual`, which here means a ceiling effect (nothing failed), not evidence that failures build up slowly.
  - **Study 2** (36 graded tasks): the local model failed 7 of 108 episodes. Its failure curves were mostly cliffs,
    **4/4** decidable curves (Wilson 95% CI **[0.510, 1.000]**), which is weak evidence from only four curves.
    Claude Haiku 5.5 solved **108/108** (ceiling, no sweep) for **$0.117**.
  - **Study 3** (96 harder tasks, levels H1–H3): difficulty grading is valid for both agents. For Claude Haiku 5.5,
    failures are **mostly cliffs: 13/15** decidable curves (Wilson 95% CI **[0.621, 0.963]**). For the local model the
    question stays open: only 5 of 25 curves were decidable, cliffs **3/5** (CI **[0.231, 0.882]**). Neither agent
    had a successful-but-unsafe episode. The paid runs cost **$7.32** of a $40 cap.
- **Scope.** One synthetic domain, two agents, task sets of 15 to 96 tasks. The results describe these agents on these
  tasks and are not claims about LLM agents in general.

## Demo: what replay can answer (synthetic, keyless, reproducible)

The demo agent is a synthetic policy with a known answer, **not an LLM**: it follows the oracle but, with probability
0.05 per turn, makes an irreversible mistake (a comment on a protected issue). In the run below the mistake happens at
turn 3. The final state only says "failed"; the replay sweep locates the step:

| checkpoint k | 0 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| P̂(success \| k) | 0.625 | 0.792 | 0.750 | 0.000 | 0.000 | 0.000 |

The largest drop, 0.750 at k = 3 → 4, is classified as a **cliff**, exactly at the irreversible step.

![synthetic recoverability curve](docs/demo/curve.svg)

The full walkthrough (task text, every tool call, every state change, the grade and the sweep table) is generated
from code and checked by the tests: see the demo section of [README.zh-TW.md](README.zh-TW.md) and `docs/demo/`.

## Key results

All numbers below come from the committed result files (`docs/results/*/summary.json`) and are checked by
`tests/test_readme_claims.py`.

### Round 1: the Core experiment (local model, 15 tasks; decisions 025, 030, 033)

- Eval baseline: **45/45** episodes succeeded (15 tasks × R = 3), trajectory-safe rate 1.000.
- Sweep: 15 sources, **480 reruns**, all successful, no cliff. Core Gate: **`gradual`**. Because nothing failed, this is
  a ceiling effect, so the project did not move on to the Advanced stage (fault injection, a second app).
- **1,739** model calls, each with its seed recorded and reconciled against the attempt manifest. The results are
  sealed (`docs/results/round1-seal.json`).

### Study 2: graded multi-step tasks (36 tasks, levels L1–L3; decisions 034, 035)

| | L1 / L2 / L3 success | Grading (L1 > L3) | RQ1: are failures mostly cliffs? |
|---|---|---|---|
| Local model | 0.972 / 0.917 / 0.917 | not valid (CI [−0.028, 0.167]) | mostly cliff, **4/4**, CI **[0.510, 1.000]**: weak evidence |
| Claude Haiku 5.5 | 1.000 / 1.000 / 1.000 | not valid | not applicable: **108/108** succeeded (ceiling) |

### Study 3: harder graded tasks, both agents (96 tasks, levels H1–H3; decision 036)

| | H1 / H2 / H3 success | Grading: H1 − H3 | RQ1: are failures mostly cliffs? | Safety |
|---|---|---|---|---|
| Local model | 0.573 / 0.385 / 0.260 | **0.312**, valid (permutation p = 0.0033) | **undeterminable**: 3/5, CI **[0.231, 0.882]** | 0 successful-but-unsafe |
| Claude Haiku 5.5 | 0.979 / 0.948 / 0.708 | **0.271**, valid (p = 0.00011) | **mostly cliff**: 13/15, CI **[0.621, 0.963]** | 0 successful-but-unsafe |

- Haiku: 12 of the 13 cliffs are behavioural (the remaining one is caused by an irreversible comment); the
  behavioural-only ratio, 12/14, is also mostly cliff. The 4 undecidable curves are all from one task family whose
  hardest level runs out of turns (a limitation registered before the evaluation).
- Local model: 20 of its 25 curves were undecidable; in 16 of them the agent rarely succeeds from *any* checkpoint
  (P̂ ≤ 0.2 everywhere), so there is no drop to detect. The frozen plan assumed that 80% of curves would be decidable.
- The two agents failed on partly different tasks, so their cliff ratios are reported side by side and are not
  compared statistically.
- Compute and cost: the local sweep used 26,642 of its 80,000-call cap; all paid Haiku calls in Study 3 cost
  **$7.32** (cap $40).

![Study 3, Claude Haiku 5.5: recoverability curves](docs/results/study3-haiku/recoverability.svg)

Full result pages (in Traditional Chinese): [round 1](docs/results/README.md), [Study 2](docs/results/study2/README.md),
[Study 2, Haiku](docs/results/study2-haiku/README.md), [Study 3, local](docs/results/study3/README.md),
[Study 3, Haiku](docs/results/study3-haiku/README.md).

## How it works

| Component | Location | What it does |
|---|---|---|
| Environment | `apps/issue_tracker/`, `env/` | Deterministic state, canonical state hash, validated and atomic snapshot / restore |
| Tasks | `tasks/` | Parameterized generators (v1: round 1; v2: Study 2; v3: Study 3) with an oracle solution and graded negatives for every task |
| Grader | `grader/` | Outcome correctness, constraint violations, forbidden actions and attempts, unnecessary mutations, tool errors, cost |
| Recorder / replay | `agents/policy.py`, `replay/` | Checkpoints carry their prefix and are verified by replay on load; a restored conversation is byte-identical to the original |
| Agent adapters | `agents/ollama.py`, `agents/baseline.py` | Ollama `/api/chat` adapter (loopback only, tool-call and JSON validation, truncation guard, model pinning); Anthropic Messages adapter |
| Failure localization | `replay/localization.py`, `replay/stats.py` | Coarse-to-fine checkpoint sweep, Wilson intervals, stratified task-level bootstrap, cliff criterion |
| Experiments | `experiments/` | Baseline and sweep runners, call caps and billing, power simulations, pre-registered analyses and reports for each study |
| GPU lock | `tools/with-gpu-lock`, `agents/ollama_serve.py` | Starts this project's own Ollama server inside a shared GPU lock and verifies on exit that it stopped and the shared model files are unchanged |

## Engineering highlights

- **Exact replay.** Restoring checkpoint *k* reproduces the environment state hash and the exact request bytes the
  model saw, so a rerun differs from the original only in the model's sampling.
- **Pre-registration as code.** Each study's protocol, task hashes, caps and statistics are written to a freeze
  record before the evaluation, and the test suite recomputes and compares it on every run. Round 1 is sealed by
  sha256.
- **Interruption-safe, resumable runs.** Long sweeps run in segments; interrupted directories are kept and adopted by
  a fixed rule (earliest finished attempt), and `--resume-from` never repeats finished work.
- **Guardrails for shared and paid compute.** The GPU wrapper never pre-empts another job's lock, and when it cannot
  confirm a clean end it reports to the user instead of deleting the lock. Local model calls are capped and logged
  with their seeds. Paid calls are reserved and settled in a journal outside the repo under a hard total cap, and
  the program never reads or stores an API key.
- **Keyless test suite.** About 11,000 tests run without a GPU or API key. CI runs `ruff` and the full suite on every
  push.

## Quick start

```bash
uv sync --extra llm                       # the llm extra also runs the SDK contract tests
uv run pytest                             # full keyless test suite (about 15 minutes on a Mac)
uv run python -m experiments.run_baseline --split dev --dry-run    # the whole baseline pipeline with an offline oracle client
uv run python -m experiments.demo_synthetic --out docs/demo --update-readme README.zh-TW.md   # regenerate the synthetic demo
uv run python -m experiments.round1_seal --check                   # verify the sealed round-1 results
```

Re-running the formal LLM experiments needs the original GPU host (local model) or an Anthropic API key (Haiku), plus
the run directories, which are not in the repository. The exact commands for every formal run are recorded in
[README.zh-TW.md](README.zh-TW.md).

> The decision logs, result pages and development notes are written in Traditional Chinese. This README is the
> English entry point.

## Research discipline

- Every design choice is a numbered decision log ([docs/decisions/](docs/decisions/), 001–037), including the
  statistical protocol, the cliff criterion and its simulated detection power, the run protocol, and the rules for
  interrupted runs.
- Metrics, task splits and models are frozen before each evaluation set is run, and each set is run once.
- Null and negative outcomes are reported as they are: a ceiling effect (round 1, Haiku in Study 2), an invalid
  difficulty grading (Study 2) and an undeterminable result (local model, Study 3).
- Development runs are reviewed one failure at a time to separate agent errors from task defects before freezing.
  Concerns that were judged not to be defects are listed in the result pages rather than silently dropped.

## Limitations

- Small evaluation sets (15, 36 and 96 tasks) in a single synthetic domain; the results are preliminary observations
  about two specific agents.
- The cliff criterion's detection power was simulated under assumed curve shapes; real shapes are unknown.
- The two agents differ in more than the model (adapter, thinking mode), so differences between them are descriptive.
- Study 3's tasks drew on Study 2's failure types and deliberately include irreversible comments, which can cause
  mechanical cliffs; this was disclosed before the evaluation, and the behavioural-only ratio is reported separately.
- For the local model in Study 3, most curves were undecidable, so its RQ1 has no conclusion.
- One evaluation episode was flagged as possibly graded too leniently (one comment contained both templates); by the
  pre-registered rule this was recorded without changing the grade.

## Status and future work

The Core is complete. By the pre-registered Gate rule the project did not move on to the Advanced stage (tool fault
injection, a second app, a failure taxonomy). Studies 2 and 3 were run as separate, pre-registered follow-ups that do
not reopen the Gate. Taking Study 3's Haiku result as grounds to reopen it would be a new decision.

## Repository layout

```text
apps/issue_tracker/   env/          deterministic environment and snapshot / restore
tasks/                              task generators v1–v3 and task format
grader/                             outcome and safety grading
agents/   replay/                   adapters, recorder, checkpoints, failure localization
experiments/                        runners, billing, analyses and reports
tools/with-gpu-lock                 shared GPU lock wrapper
docs/decisions/                     decision logs (Traditional Chinese)
docs/results/                       committed results, figures and summary.json files
tests/                              keyless test suite
```

## License

The original code is released under the BSD-3-Clause license: see [LICENSE](LICENSE).
