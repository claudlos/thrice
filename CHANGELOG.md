# Changelog

All notable changes to Thrice are recorded here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
semver is "best-effort" since the project is pre-1.0.

## [Unreleased]

### Added (research-grounded wave)
Three modules driven by specific current research on agent-harness design:

- `modules/cache_optimizer.py` — `PrefixGuard` scans for cache-invalidating
  dynamic values (timestamps, UUIDs, session IDs, tool-set churn) and
  `CacheTracker` records per-turn hit rate with a rolling-window
  regression detector.  Grounded in the Manus engineering blog
  (*"KV-cache hit rate is the single most important metric for a
  production-stage AI agent"*) and *"Don't Break the Cache: An Evaluation
  of Prompt Caching for Long-Horizon Agentic Tasks"* (arXiv:2601.06007)
  which measured 45-80 % cost reduction from correct caching.  19 tests.
- `modules/task_scratchpad.py` — the **recitation pattern** from the Manus
  blog: a bounded, revisable todo list that the agent re-renders at the
  end of every prompt so the goal stays in recent context across 50+
  tool-use loops.  Append-only revision log, atomic save, depth + item
  + text caps, collapse-completed rendering.  19 tests.
- `modules/subagent_dispatch.py` — fresh-context delegation primitive from
  the Advanced Context Engineering guide (humanlayer, 2025).  Pluggable
  `Runner` protocol (in-process / LLM / subprocess); hard caps on tokens,
  wall-clock, and output size; canonical `search` / `summarize` /
  `inspect` task patterns; ``run_many`` fan-out.  Integrates with
  `cost_estimator` for budget gating.  15 tests.

### Changed / Fixed (post-ship sweep)
- **Lint**: whole-tree ruff sweep. 98 module-side + 85 test-side import-order
  / unused-import / ambiguous-name / `B007` / `B905` / `B904` issues
  fixed.  CI `lint` step tightened from advisory to blocking.
- **Real bug**: `update.py` referenced `os.environ.get(...)` in
  `find_hermes_agent` but never imported `os` — would have raised
  `NameError` on every invocation.  Fixed + verified.
- **pytest collection warnings**: 8 business classes in `modules/`
  (``TestRunner``, ``TestResult``, ``TestFixLoop``, etc.) were being
  picked up as pytest test classes because of their names.  Added
  ``__test__ = False`` after their docstrings so collection is clean.
- **Integration**: `modules/auto_commit.py :: on_file_edit` now:
  - runs `secret_scanner.scan_diff` over staged hunks and refuses to
    commit when high/medium severity findings fire;
  - builds commit messages via `conventional_commit.suggest_type` when
    the helper is importable.  Graceful ``ImportError`` degradation
    means neither integration is mandatory.  5 regression tests added.
- **Windows robustness**: `conventional_commit.suggest_type` now
  normalises path separators so file paths that come through as
  ``tests\foo.py`` (native Windows) are classified the same as
  ``tests/foo.py``.
- **Concurrency**: `cost_estimator.record` now serialises writes with an
  in-process `threading.Lock`; 4-thread × 50-record test verifies no
  lines are lost or torn.
- **Resource leak**: `lsp_bridge` now spawns a dedicated stderr-drain
  thread so chatty servers can't fill the pipe buffer and deadlock the
  reader thread.
- **REFINEMENT.md**: new "SM-3 · Bisector" section, CI prose updated to
  mention the third spec and the coverage sanity check.
- **CI**: `py_compile` now compiles `specs/tla/_check_coverage.py`;
  `lint` job runs `ruff check` on `tests/` too (advisory).

### Added
- **Eight more coding-improvement modules** completing the 10-item sweep
  from the previous audit round:
  - `modules/conventional_commit.py` — Conventional Commits validator +
    generator + auto-type-suggestion from a diff.  24 tests.
  - `modules/secret_scanner.py` — regex + Shannon-entropy scan for AWS /
    GitHub / Slack / Stripe / OpenAI / Anthropic / JWT / PEM / password
    literals in text, diffs, or files.  Masks matches before reporting.
    24 tests.
  - `modules/doctest_runner.py` — project-wide doctest runner with
    structured failure records ``(file, line, want, got)``.  11 tests.
  - `modules/diff_preview.py` — pre-apply syntax validator (Python AST,
    JSON, TOML, YAML, universal conflict-marker + bracket-balance check).
    Refuses to let a broken edit hit disk.  Includes an in-memory
    unified-diff applier for patch-level preview.  16 tests.
  - `modules/semantic_diff.py` — AST-level Python diff.  Ignores
    whitespace, comments, import reorder; detects
    added / removed / renamed / signature-changed / body-changed /
    decorator-changed symbols at qualname granularity.  17 tests.
  - `modules/trace_capture.py` — stack + redacted-locals capture on
    test / callable failure.  Safe-`repr` with length caps; sensitive
    names (``password``, ``token``, ``api_key``, …) auto-redacted.
    12 tests.
  - `modules/cost_estimator.py` — JSONL-backed history with kNN
    (exact / prefix) + feature-fallback + static fallback; returns p50 +
    p95 token / iteration / wall-clock budgets with a recency-weighted
    percentile estimator.  11 tests.
  - `modules/lsp_bridge.py` — synchronous stdio JSON-RPC 2.0 LSP client.
    Supports ``definition``, ``references``, ``hover``,
    ``documentSymbols``, ``rename`` against pyright / pylsp /
    rust-analyzer / gopls / tsserver.  14 tests (driven by a tiny fake
    LSP server so CI doesn't need a real language server installed).
- **Two new modules** driven by the audit's "coding improvements" section:
  - `modules/build_loop.py` — structured build diagnostics (`cargo` /
    `go build` / `tsc --noEmit` / `gcc` / `py_compile`).  Parsers are
    pure and individually unit-tested; 19 tests covering Rust, Go,
    TypeScript, C, and Python diagnostics plus end-to-end build runs.
  - `modules/regression_bisector.py` — `git bisect` driver with per-step
    and total timeouts.  Corresponds to SM-3 in `specs/tla/Bisector.tla`
    (6 states, 8 transitions, 5 safety invariants, 3 liveness properties).
- **SM-3 TLA+ specification** (`specs/tla/Bisector.tla`,
  `MCBisector.tla`, `MCBisector.cfg`).  `run_tlc.sh` and the TLA+ CI
  matrix extended; the variable-coverage check also includes the new
  spec.
- **Formal TLA+ specifications** for SM-1 (cron job lifecycle) and SM-2
  (agent request/response loop):
  - `specs/tla/CronJob.tla` — 12 transition actions, 8 safety invariants,
    4 liveness properties.
  - `specs/tla/AgentLoop.tla` — 16 transition actions, 8 safety invariants,
    4 liveness properties.
  - `specs/tla/run_tlc.sh` — self-contained TLC runner that auto-downloads
    `tla2tools.jar`.
  - `specs/tla/_check_coverage.py` — sanity check that every variable is
    covered in every action.
  - `.github/workflows/tla.yml` — TLC model checks run in CI on every push.
  - `specs/REFINEMENT.md` — simulation relation between the Python state
    machines and their TLA+ specifications.
- `tests/refinement_tests/test_thrice_sm_refinement.py` — 14 Python tests
  driving the concrete state machines through the abstract transition
  tables and TLA+ invariants.
- `.github/workflows/ci.yml` — matrix of `py_compile`, `pytest`, `ruff`,
  and installer dry-run jobs across Python 3.10–3.13 on Linux + Windows.
- `pyproject.toml` — PEP 621 packaging metadata, `[project.optional-dependencies]`
  for test and dev extras, pytest + ruff config.
- `SECURITY.md`, `CONTRIBUTING.md`, `CHANGELOG.md`.
- `.github/pull_request_template.md` and issue templates under
  `.github/ISSUE_TEMPLATE/`.
- `tests/test_state_machine.py` — 16 direct tests on the generic
  `StateMachine` framework (previously only exercised transitively).
- `tests/conftest.py` — new `requires_hermes` pytest marker with auto-skip
  when `hermes-agent` isn't on `PYTHONPATH`.
- Regression tests for the `@file:` path-traversal guard in
  `context_mentions._resolve_file`.

### Changed
- `modules/cron_state_machine.py :: mark_success()` — context update made
  atomic with the transition so `TerminalNoNextRun` / `SCHEDULED_NO_NEXT_RUN`
  invariants hold during `apply()`.  Bug caught by the TLA+ refinement
  test when it was first wired up.
- `modules/agent_loop_state_machine.py :: build_request()` — iteration
  counter now incremented *after* the guard check (matches
  `AgentLoop.tla :: BuildRequest`).  Previously the pre-increment made
  `has_budget` off-by-one, so `iteration_budget=1` could never actually
  succeed.
- `modules/agent_loop_state_machine.py :: retry_api()` — removed
  `result.accepted` dereference (attribute doesn't exist on
  `TransitionRecord`); `apply()` raises on failure so the increment is
  always safe.
- `modules/context_mentions.py :: _resolve_file()` — path-traversal guard
  that keeps `@file:` lookups inside the working directory (rejects
  absolute paths, `..` segments, and symlink escapes).
- `modules/skill_index_cache.py` — now a one-line re-export of
  `modules/agent/skill_index_cache.py` (previously a byte-identical
  duplicate).  `modules/agent/__init__.py` added so the package
  resolves cleanly.
- `install.py` — `cron_state_machine.py` added to `STANDALONE_MODULES`.
  Module count updated to 39.
- `tests/test_skill_cache.py::test_mtime_change_busts_cache` — explicit
  `os.utime()` bump so the test is deterministic on Windows NTFS where
  mtime resolution is coarser than the write-and-read interval.
- `tests/test_auto_commit.py::test_run_git_failure` — sets
  `GIT_CEILING_DIRECTORIES` so `git status` in a temp dir actually
  fails (previously walked up to the user's home-dir repo on
  developer machines).
- `tests/test_tool_chain_analysis.py :: tmp_db` fixture — tolerates the
  Windows case where SQLite's WAL file holds the database open briefly
  after `connect() as conn:` exits.
- Several test files — import path normalized to prefer `modules/` over
  the legacy `new-files/` location.
- Hermes-dependent test files (`test_cron_state_machine`,
  `test_integration_smoke`, `tests/invariant_tests/test_cron_invariants`,
  `tests/refinement_tests/test_cron_refinement`,
  `tests/property_tests/test_process_lifecycle`,
  `tests/property_tests/test_message_integrity`) now skip at import time
  when the live Hermes API isn't available.
- README — license line corrected to "MIT" (previously claimed "Same
  license as Hermes Agent" while `LICENSE` was already MIT); module and
  test count badges updated to reflect current reality; new "Formal
  specifications" section links to the TLA+ specs.

### Removed
- `modules/agent_loop_refactor.py` — dead code (not installed, not
  imported, not tested; was pre-release documentation of the refactor
  approach).

### Security
- Path-traversal guard added to `@file:` mention resolution
  (`modules/context_mentions.py`).  Previously an attacker-controlled
  prompt containing `@file:/etc/passwd` or `@file:../secrets` could
  read arbitrary files the agent had permission to open.

## [0.1.0] — 2026-04-04

### Added
- Initial release: 38 modules, 15 patches, 1338 tests.
