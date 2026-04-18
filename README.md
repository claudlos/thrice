<p align="center">
  <img src="assets/banner.png" alt="Thrice — Hermes Trismegistus" width="100%">
</p>

<h1 align="center">Thrice</h1>

<p align="center">
  <em>"Thrice Great"</em> — <a href="https://github.com/NousResearch/hermes-agent">Hermes</a> 
</p>

<p align="center">
  <img src="https://img.shields.io/badge/modules-52-blue" alt="52 modules">
  <img src="https://img.shields.io/badge/tests-1456_passing-green" alt="1456 passing without Hermes; 7 skipped until Hermes is installed">
  <img src="https://img.shields.io/badge/patches-15-orange" alt="15 patches">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://github.com/claudlos/thrice/actions/workflows/tla.yml/badge.svg" alt="TLA+ verified">
</p>

---

**Thrice** is a drop-in improvement suite for [Hermes Agent](https://github.com/NousResearch/hermes-agent). It adds 52 standalone Python modules covering smarter tool use, better context management, formal state machine hardening, multi-agent coordination, and research-grade features — all with graceful fallbacks. Delete any module file and Hermes instantly reverts to its original behavior.

## Quick Start

```bash
git clone https://github.com/claudlos/thrice.git
cd thrice

# Preview what will happen (no changes made)
python3 install.py --dry-run

# Install everything
python3 install.py

# Or install only standalone modules (safe, no patching)
python3 install.py --modules-only
```

## What's Inside

### Core Improvements

| Module | What it does |
|--------|-------------|
| `tool_alias_map.py` | Resolves hallucinated tool names (bash→terminal, grep→search_files) |
| `adaptive_compression.py` | Scales context compression threshold by model window size |
| `smart_truncation.py` | Intelligent output truncation preserving errors and structure |
| `comment_strip_matcher.py` | Fuzzy matching that strips comments before comparing |
| `structured_errors.py` | Typed error hierarchy with recovery hints |
| `debugging_guidance.py` | System prompt injection for structured debugging |
| `context_mentions.py` | @file, @symbol, @diff mention resolution in user messages |
| `context_optimizer.py` | Information-theoretic context scoring and pruning |
| `prompt_algebra.py` | Composable prompt building with algebraic operations |
| `repo_map.py` | Repository structure maps with symbol-level detail |
| `auto_commit.py` | Auto git commits after edits with /undo rollback |
| `agent/skill_index_cache.py` | Cached skill lookups for faster prompt building |

### Agentic Workflow

| Module | What it does |
|--------|-------------|
| `reproduce_first.py` | Reproduce-before-fix debugging — confirms the bug before patching |
| `test_fix_loop.py` | Iterative test-fix loop with multi-language parsers (pytest, jest, cargo, go) |
| `build_loop.py` | Structured build diagnostics for cargo / go / tsc / gcc / py_compile |
| `regression_bisector.py` | Scripted `git bisect` with step + total timeouts (SM-3 in TLA+) |
| `context_gatherer.py` | Smart context gathering — reads imports, tests, and reverse deps before editing |
| `edit_format.py` | Selects optimal edit format per file size and change scope |
| `error_recovery.py` | Classifies errors and applies graduated recovery strategies |
| `change_impact.py` | Analyzes what breaks when you change a function signature |
| `tool_call_batcher.py` | Detects independent tool calls and suggests parallel execution |
| `project_memory.py` | Persistent project context across sessions (like CLAUDE.md) |
| `conversation_checkpoint.py` | Save and restore conversation state for long tasks |
| `agent_loop_components.py` | Modular agent loop: ToolDispatcher, RetryEngine, CostTracker |
| `silent_error_audit.py` | Detects silent error swallowing + module health dashboard |

### Coding Quality Gates

| Module | What it does |
|--------|-------------|
| `diff_preview.py` | Syntax-validate a proposed edit (Python / JSON / YAML / TOML / generic) before it hits disk |
| `semantic_diff.py` | AST-level diff for Python — ignores whitespace, comments, import reorder; detects renames / signature / body / decorator changes |
| `trace_capture.py` | Capture stack + redacted local variables on test failures for the agent's next turn |
| `doctest_runner.py` | Run docstring examples project-wide with structured failure records |
| `secret_scanner.py` | Regex + entropy pre-commit scan; flags AWS / GitHub / OpenAI / Anthropic / JWT / PEM / password literals |
| `conventional_commit.py` | Validate + auto-generate Conventional-Commits messages from a diff |
| `cost_estimator.py` | Predict p50 / p95 token / iteration / wall-clock cost from historical telemetry |
| `lsp_bridge.py` | Stdio JSON-RPC 2.0 LSP client: `definition` · `references` · `hover` · `documentSymbols` · `rename` |

### Context & Cache (research-grounded)

Grounded in the Manus context-engineering blog and *"Don't Break the Cache"*
(arXiv:2601.06007), which measured 45-80 % cost reduction from correct KV-cache
preservation in agent loops.

| Module | What it does |
|--------|-------------|
| `cache_optimizer.py` | `PrefixGuard` detects cache-invalidating mutations (timestamps, UUIDs, tool-set churn) before a request goes out; `CacheTracker` records per-turn hit rate, estimated $ savings, and flags regressions |
| `task_scratchpad.py` | Recitation pattern — bounded, revisable todo list that gets re-rendered at the end of every prompt so the objective stays in recent context across 50+ tool-use loops |
| `subagent_dispatch.py` | Fresh-context delegation primitive with token/time/output caps, canonical `search` / `summarize` / `inspect` patterns, and a pluggable runner so tests don't need a live LLM |

### Research & Formal Methods

| Module | What it does |
|--------|-------------|
| `tool_selection_model.py` | ML-based tool prediction from task description |
| `tool_chain_analysis.py` | Causal analysis of tool usage patterns |
| `verified_messages.py` | HMAC-signed message integrity for multi-agent |
| `consensus_protocol.py` | Multi-agent consensus voting protocol |
| `token_budgeting.py` | Predictive token usage and iteration budgeting |
| `self_reflection.py` | Post-edit self-review before proceeding |
| `self_improving_tests.py` | Auto-generated regression tests from session data |
| `skill_verifier.py` | Skill integrity verification |

### State Machine Infrastructure

| Module | What it does |
|--------|-------------|
| `state_machine.py` | Generic state machine framework with invariant checking |
| `agent_loop_state_machine.py` | Formal state machine for the Hermes agent loop |
| `agent_loop_shadow.py` | Shadow testing — observes state transitions without affecting behavior |
| `enforcement.py` | Three enforcement modes: development, production, testing |
| `hermes_invariants.py` | Runtime invariant checks across 6 subsystems |
| `invariants_unified.py` | Unified invariant system with temporal checks |
| `test_lint_loop.py` | Lower-level test/lint iteration primitives |

### 15 Patches to Hermes Core

Patches wire the modules into Hermes at key integration points (`run_agent.py`, `cron/jobs.py`, `tools/`, `gateway/`). Every patch follows the same pattern:

```python
try:
    from some_module import SomeClass
    # use it
except ImportError:
    pass  # graceful fallback to original behavior
```

## Environment Variables

Some features are gated (off by default):

```bash
export HERMES_SM_SHADOW=true       # Shadow state machine observer
export HERMES_AUTO_COMMIT=true     # Auto git commits after file edits
export HERMES_SELF_TEST=true       # Self-improving test generation
export HERMES_PREDICT_BUDGET=true  # Token budget prediction
```

Everything else activates automatically when the module file is present.

## After Hermes Updates

Standalone modules survive updates. Patches may be overwritten. To re-apply:

```bash
python3 update.py              # re-apply everything
python3 update.py --modules-only  # just modules (always safe)
python3 update.py --dry-run    # preview first
```

## Uninstall

```bash
python3 uninstall.py            # remove everything, restore backups
python3 uninstall.py --dry-run  # preview
```

Or just delete any module file — the `try/except ImportError` fallback handles it.

## Development

```bash
# Run the module-level test suite (no Hermes required)
pip install pytest hypothesis
PYTHONPATH=modules pytest tests/ -q \
    --ignore=tests/test_cron_state_machine.py \
    --ignore=tests/test_hermes_invariants.py \
    --ignore=tests/test_integration_smoke.py

# Full suite (requires Hermes on PYTHONPATH)
PYTHONPATH="$HOME/.hermes/hermes-agent:modules:." pytest tests/ -q

# Syntax check all modules
for f in modules/*.py modules/agent/*.py; do python3 -m py_compile "$f"; done

# Formal verification (requires Java 11+)
./specs/tla/run_tlc.sh                 # both specs
./specs/tla/run_tlc.sh cron            # just SM-1
./specs/tla/run_tlc.sh agent           # just SM-2
```

## Formal specifications

The two core state machines (`cron_state_machine.py`, `agent_loop_state_machine.py`)
are specified in TLA+ and model-checked by TLC on every push:

| Spec                       | Models                                 |
|----------------------------|----------------------------------------|
| [`specs/tla/CronJob.tla`](specs/tla/CronJob.tla)     | SM-1 — cron job lifecycle (7 states, 12 transitions)   |
| [`specs/tla/AgentLoop.tla`](specs/tla/AgentLoop.tla) | SM-2 — agent request/response loop (11 states, 16 transitions) |
| [`specs/tla/Bisector.tla`](specs/tla/Bisector.tla)   | SM-3 — regression bisector lifecycle (6 states, 8 transitions) |

See [`specs/tla/README.md`](specs/tla/README.md) for what's checked (invariants +
liveness properties) and [`specs/REFINEMENT.md`](specs/REFINEMENT.md) for the
simulation relation between TLA+ actions and Python methods.

TLC runs in CI — see [`.github/workflows/tla.yml`](.github/workflows/tla.yml).

## How It Works

Thrice follows three principles:

1. **Graceful degradation** — every module is wrapped in `try/except ImportError`. Remove the file and Hermes works exactly as before.

2. **No forking** — Thrice doesn't fork Hermes. It layers improvements on top via standalone modules + surgical patches. When Hermes updates, run `update.py` to re-apply.

3. **Mathematically rigorous** — the two core state machines have full TLA+ specifications (`specs/tla/CronJob.tla`, `specs/tla/AgentLoop.tla`), model-checked by TLC in CI against nine safety invariants and four liveness properties apiece, with property-based tests (Hypothesis) for the Python side. The specs caught two real atomicity bugs in the Python implementation (pre-transition context clear in `mark_success`, pre-guard iteration increment in `build_request`) when they were first wired up.

## License

MIT — see [`LICENSE`](LICENSE).

Thrice is a layer on top of [Hermes Agent](https://github.com/NousResearch/hermes-agent)
but is distributed under its own MIT license.  Hermes itself is licensed
separately; consult its repository before redistribution.
