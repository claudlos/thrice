# Refinement: Python state machines ↔ TLA+ specifications

This document defines the **simulation relation** `R` between the concrete
Python state machines in `modules/` and the abstract TLA+ specifications
in `specs/tla/`.  Its purpose is operational:

1. Tell a reader which Python method implements which TLA+ action.
2. Tell a reviewer which Python invariants the TLC run already proves.
3. Make the refinement claim falsifiable — when code drifts, the
   property-based tests in `tests/refinement_tests/` should catch it.

A concrete run of the Python class *refines* the TLA+ spec if, for every
concrete step, projecting the concrete variables through `R` gives a step
that is allowed by the TLA+ `Next` relation (or is stuttering).

---

## 1. SM-1 · Cron Job (`CronJob.tla` ↔ `modules/cron_state_machine.py`)

### State mapping

The Python enum is exactly the TLA+ `States` set — the relation is the
identity:

| Python `CronJobStateMachine.state` | TLA+ `state` |
|------------------------------------|--------------|
| `"nonexistent"`                    | `"nonexistent"` |
| `"scheduled"`                      | `"scheduled"`   |
| `"running"`                        | `"running"`     |
| `"paused"`                         | `"paused"`      |
| `"completed"`                      | `"completed"`   |
| `"failed"`                         | `"failed"`      |
| `"removed"`                        | `"removed"`     |

### Variable projection `π`

| Python attribute (in `.context`)       | TLA+ variable         | Notes |
|----------------------------------------|-----------------------|-------|
| `recurring: bool`                      | `recurring`           | direct |
| `repeat_total: Optional[int]`          | `repeatTotal`         | `None → 0` |
| `runs_completed: int`                  | `runsCompleted`       | direct |
| `runs_left: Optional[int]`             | `runsLeft`            | `None → 0` when `repeatTotal=0` |
| `retry_count: int`                     | `retryCount`          | direct |
| `max_retries: int`                     | `maxRetries`          | direct |
| `enabled: bool`                        | `enabled`             | direct |
| `next_run_at: Optional[datetime]`      | `nextRunDue`          | `now >= next_run_at → TRUE` |
| `tick_pending: bool`                   | `tickPending`         | direct |
| `removed: bool`                        | `removedFlag`         | direct |
| stale lock observation                 | `staleLock`           | external signal |
| `last_error: Optional[str]`            | `lastError`           | `None → FALSE` |

### Action mapping

| Python method                | TLA+ action                          |
|------------------------------|--------------------------------------|
| `create(...)`                | `Create(recurring, repeat_total)`    |
| `tick(now)`                  | `Tick`                               |
| `mark_success(...)` (rec.)   | `MarkSuccessRecurring`               |
| `mark_success(...)` (done)   | `MarkSuccessDone`                    |
| `mark_failure(...)`          | `MarkFailure`                        |
| `retry(...)`                 | `Retry`                              |
| `pause()`                    | `Pause`                              |
| `resume(...)`                | `Resume`                             |
| `trigger()`                  | `Trigger`                            |
| `remove()`                   | `Remove`                             |
| `recover()`                  | `Recover`                            |
| wall-clock advance           | `BecomeDue` (environment)            |
| worker crash                 | `LockGoesStale` (environment)        |

### Invariants proved by TLC that the Python class relies on

| TLA+ invariant          | Python invariant function                          |
|-------------------------|----------------------------------------------------|
| `ValidState`            | `_invariant_valid_state`                           |
| `RunningNotDeleted`     | `_invariant_running_not_deleted`                   |
| `PausedNeverFires`      | `_invariant_paused_never_fires`                    |
| `RepeatConsistency`     | `_invariant_repeat_consistency`                    |
| `TerminalNoNextRun`     | `_invariant_completed_no_next_run` (+extension)    |
| `RetryBounded`          | implicit in `_guard_retry`                         |
| `CompletedHasRun`       | (TLA+ only — not currently asserted in Python)     |

### Liveness proved by TLC

- `RunningMakesProgress` — forbids "stuck running" absent a real crash.
- `FailureIsRetried` — a failed job with budget eventually moves on.
- `OneShotTerminates` — `~recurring /\ running ~> completed|failed|removed`.
- `RemovedAbsorbs` — `removed` is absorbing.

---

## 2. SM-2 · Agent Loop (`AgentLoop.tla` ↔ `modules/agent_loop_state_machine.py`)

### State mapping

Python `AgentLoopState` enum values map 1:1 to the TLA+ state strings.
See the header of `agent_loop_state_machine.py` for the enum, and the
`CONSTANTS` block at the top of `AgentLoop.tla` for the matching strings.

### Variable projection `π`

The Python class stores loop-local state inside `AgentLoopContext`.
The projection into TLA+ variables is:

| Python `AgentLoopContext`            | TLA+ variable              |
|--------------------------------------|----------------------------|
| `len(messages) > 0`                  | `messagesNonEmpty`         |
| `iterations_used`                    | `iterationsUsed`           |
| `retry_count`                        | `retryCount`               |
| `bool(api_params)`                   | `apiParamsBuilt`           |
| `bool(pending_tool_calls)`           | `hasToolCalls`             |
| `needs_continue` property            | `needsContinue`            |
| `context_full` property              | `contextFull`              |
| `interrupt_flag`                     | `interruptFlag`            |
| invariant `_inv_system_prompt_preserved` | `systemPromptPreserved` |

### Action mapping

| Python method                               | TLA+ action                        |
|---------------------------------------------|------------------------------------|
| `receive_message(...)`                      | `ReceiveMessage`                   |
| `build_request(...)`                        | `BuildRequest`                     |
| `exhaust_budget()`                          | `ExhaustBudget`                    |
| `receive_response(...)`                     | `ReceiveResponse`                  |
| `recover_error(...)`                        | `RecoverError`                     |
| `dispatch_tools()`                          | `DispatchTools`                    |
| `process_text_response()`                   | `ProcessTextResponse`              |
| `tool_complete(...)`                        | `ToolComplete`                     |
| `compress(...)`                             | `Compress`                         |
| `continue_after_compression()`              | `ContinueAfterCompression`         |
| `continue_generation()`                     | `ContinueGeneration`               |
| `return_result()` (turn-done branch)        | `ReturnResultTurnDone`             |
| `return_result()` (retry-exhausted branch)  | `ReturnResultRetriesExhausted`     |
| `retry_api()`                               | `RetryAPI`                         |
| `interrupt()`                               | `Interrupt`                        |
| signal handler sets `interrupt_flag`        | `RaiseInterrupt` (environment)     |

### Invariants proved by TLC that the Python class relies on

| TLA+ invariant              | Python invariant function              |
|-----------------------------|----------------------------------------|
| `MessagesNonEmptyOnCall`    | `_inv_messages_non_empty_before_call`  |
| `BudgetMonotonic`           | `_inv_budget_monotonic`                |
| `RetryBounded`              | `_inv_retry_bounded`                   |
| `RetryResetsOnResponse`     | `_inv_retry_resets_on_success`         |
| `SystemPromptPreserved`     | `_inv_system_prompt_preserved`         |
| `ApiParamsReadyOnCall`      | implicit in `_guard_api_params_valid`  |
| `ValidState`                | underlying `StateMachine` enforces     |

### Liveness proved by TLC

- `Termination` — the loop always exits (given finite budget).
- `InterruptLiveness` — the Python `Interrupt` path is always reachable
  from any non-terminal state.
- `BudgetEventuallyFires` — running out of iterations forces a terminal.
- `TerminalStaysTerminal` — terminal states are absorbing.

---

## 3. SM-3 · Bisector (`Bisector.tla` ↔ `modules/regression_bisector.py`)

### State mapping

Python `Bisector._state` strings map identity-wise to the TLA+ enum:

| Python state     | TLA+ state         |
|------------------|--------------------|
| `"idle"`         | `"idle"`           |
| `"testing"`      | `"testing"`        |
| `"narrowing"`    | `"narrowing"`      |
| `"found"`        | `"found"`          |
| `"completed"`    | `"completed"`      |
| `"aborted"`      | `"aborted"`        |

### Variable projection `π`

The Python class tracks `BisectStep` records and the input config; the
abstract model compresses this into an integer interval `[lo, hi]` over
the commit chain plus a counter.

| Python                                  | TLA+ variable                     |
|-----------------------------------------|-----------------------------------|
| `config.good` position                  | `lo` (index of current good bound)|
| `config.bad` position                   | `hi` (index of current bad bound) |
| actual regressing commit                | `target` (fixed by environment)   |
| `len(steps)`                            | `steps`                           |

### Action mapping

| Python method / code path                   | TLA+ action                           |
|---------------------------------------------|---------------------------------------|
| `Bisector.run()` initial handshake          | `Start`                               |
| `_run_test` outcome = `good`                | `TestMidpointGood`                    |
| `_run_test` outcome = `bad`                 | `TestMidpointBad`                     |
| post-step interval width = 1                | `Converge`                            |
| post-step interval width > 1                | `Continue`                            |
| final `git bisect reset`                    | `Finish`                              |
| `BisectError` raised anywhere               | `Abort` / `StepLimitAbort`            |

### Invariants proved by TLC the Python class relies on

| TLA+ invariant              | Python enforcement                       |
|-----------------------------|------------------------------------------|
| `ValidState`                | `self._state: BisectState` is a Literal  |
| `IntervalContainsTarget`    | driven by `git bisect`; assumed correct  |
| `StepsBounded`              | `BisectConfig.max_steps`                 |
| `FoundIsSingleton`          | `_extract_first_bad_commit` pulls one sha|
| `TerminalAbsorbing`         | no code sets state out of a terminal     |

### Liveness proved by TLC

- `Termination` — every run lands in `completed` or `aborted`.
- `ConvergesIfNoAbort` — absent a real error, we reach `found`.
- `TerminalStaysTerminal`.

---

## 4. How CI proves refinement

On every push to `main` the workflow `.github/workflows/tla.yml` runs:

1. `python -m py_compile modules/**/*.py` — syntax check on all modules.
2. `specs/tla/_check_coverage.py` — every declared variable is primed or
   `UNCHANGED` in every TLA+ action across all three specs.
3. `./specs/tla/run_tlc.sh` — TLC model checks `MCCronJob`,
   `MCAgentLoop`, and `MCBisector` against every invariant and liveness
   property listed above, at the bounds in `MC*.cfg`.
4. `pytest tests/refinement_tests/ -q` — Python-side simulation tests
   that drive each state machine through scripted operation sequences
   and assert every reached state is in the abstract `States` set and
   every transition is allowed by the abstract transition table.

If any of these fails, the PR is blocked.

## 5. What is *not* yet proved

These are honest gaps worth tracking:

- The tool-call fan-out in `EXECUTING_TOOLS` is abstracted as a single
  `ToolComplete` step.  A per-tool spec (bounded concurrency, partial
  failure) would be a good follow-up.
- `Compress` does not model *how* compression preserves the system
  prompt — we only assert the flag.  A message-sequence level spec
  would make INV-A8 genuine rather than axiomatic.
- There is no cross-job concurrency spec for cron.  The Python
  `check_all_cron_invariants` function already asserts "at most one
  running job" as an optional rule; a multi-job TLA+ spec would turn it
  into a proof.
