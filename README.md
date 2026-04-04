<p align="center">
  <img src="assets/banner.png" alt="Thrice — Hermes Trismegistus" width="100%">
</p>

<h1 align="center">Thrice</h1>

<p align="center">
  <em>"Thrice Great"</em> — making <a href="https://github.com/livetheoogway/hermes">Hermes</a> even greater than it already is.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/modules-38-blue" alt="38 modules">
  <img src="https://img.shields.io/badge/tests-1351-green" alt="1351 tests">
  <img src="https://img.shields.io/badge/patches-15-orange" alt="15 patches">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
</p>

---

**Thrice** is a drop-in improvement suite for the [Hermes AI Agent](https://github.com/livetheoogway/hermes). It adds 38 standalone Python modules covering smarter tool use, better context management, formal state machine hardening, multi-agent coordination, and research-grade features — all with graceful fallbacks. Delete any module file and Hermes instantly reverts to its original behavior.

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
| `context_gatherer.py` | Smart context gathering — reads imports, tests, and reverse deps before editing |
| `edit_format.py` | Selects optimal edit format per file size and change scope |
| `error_recovery.py` | Classifies errors and applies graduated recovery strategies |
| `change_impact.py` | Analyzes what breaks when you change a function signature |
| `tool_call_batcher.py` | Detects independent tool calls and suggests parallel execution |
| `project_memory.py` | Persistent project context across sessions (like CLAUDE.md) |
| `conversation_checkpoint.py` | Save and restore conversation state for long tasks |
| `agent_loop_components.py` | Modular agent loop: ToolDispatcher, RetryEngine, CostTracker |
| `silent_error_audit.py` | Detects silent error swallowing + module health dashboard |

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
# Run the test suite
pip install pytest hypothesis
PYTHONPATH="$HOME/.hermes/hermes-agent:." pytest tests/ -q

# Syntax check all modules
for f in modules/*.py modules/agent/*.py; do python3 -m py_compile "$f"; done
```

## How It Works

Thrice follows three principles:

1. **Graceful degradation** — every module is wrapped in `try/except ImportError`. Remove the file and Hermes works exactly as before.

2. **No forking** — Thrice doesn't fork Hermes. It layers improvements on top via standalone modules + surgical patches. When Hermes updates, run `update.py` to re-apply.

3. **Mathematically rigorous** — core state machines have TLA+ specifications, property-based tests (Hypothesis), and runtime invariant checking. The formal specs caught 5 race conditions and proved deadlock freedom.

## License

Same license as [Hermes Agent](https://github.com/livetheoogway/hermes).
