# Thrice · TLA+ specifications

This directory holds the formal TLA+ specifications for the state machines
that Hermes ships via Thrice.  They are *not* decorative - they are run in
CI on every commit, and they document the transition rules, invariants,
and liveness properties that the Python implementation refines.

| File | Describes |
|------|-----------|
| `CronJob.tla`      | SM-1 — per-job cron lifecycle (`modules/cron_state_machine.py`). |
| `AgentLoop.tla`    | SM-2 — Hermes request/response loop (`modules/agent_loop_state_machine.py`). |
| `MCCronJob.tla`    | TLC harness that instantiates `CronJob` with small bounds. |
| `MCAgentLoop.tla`  | TLC harness that instantiates `AgentLoop` with small bounds. |
| `MCCronJob.cfg`    | Constants, invariants, and properties to check for SM-1. |
| `MCAgentLoop.cfg`  | Constants, invariants, and properties to check for SM-2. |
| `run_tlc.sh`       | Thin wrapper that downloads `tla2tools.jar` if missing and runs TLC. |

## What is checked

Both specs are model-checked by TLC on every CI run for:

- `TypeOK` — every reachable state has well-typed variables.
- State-enum closure (`ValidState`).
- Transition guards — no transition reaches an invalid state.
- The Python-side invariants (`INV-*`) encoded as TLA+ state predicates.
- Liveness — defined in `PROPERTIES`, checked with fairness constraints
  on progress actions.

## Running locally

```bash
# one shot, both specs, small bounds (~seconds)
./specs/tla/run_tlc.sh

# only one spec
./specs/tla/run_tlc.sh cron
./specs/tla/run_tlc.sh agent

# larger state space (raise constants in MC*.cfg first)
TLC_WORKERS=auto ./specs/tla/run_tlc.sh
```

Requirements:

- Java 11+ on `PATH`.
- Either `curl` or `wget` (only needed on the first run; the script downloads
  `tla2tools.jar` into this directory and caches it).

## Bounds

TLC is a bounded model checker.  The defaults in `MCCronJob.cfg`
(`MaxRepeat = 2`, `MaxRetries = 2`) and `MCAgentLoop.cfg`
(`IterationBudget = 3`, `MaxRetries = 2`) are deliberately small so the
checker finishes in seconds on CI.  For a heavier sweep (exhaustive
reachability to depth ~10) raise those constants; TLC runtime grows
roughly quadratically in the numeric bounds.

## Invariants that are genuinely enforced

**CronJob**

- `ValidState` — state variable never leaves the enum.
- `RunningNotDeleted` — no job is simultaneously `running` *and*
  `removedFlag` (prevents a tombstoned live worker).
- `PausedNeverFires` — paused jobs never carry a pending tick.
- `RepeatConsistency` — for bounded recurring jobs,
  `runsCompleted + runsLeft = repeatTotal` at every reachable state,
  and a `completed` job has `runsLeft = 0`.
- `TerminalNoNextRun` — `completed` and `removed` jobs never have a
  live `nextRunDue`.
- `RunningIsEnabled` — if the state is `running`, `enabled` is true.
- `RetryBounded` — `retryCount <= maxRetries + 1`.
- `CompletedHasRun` — a `completed` job has `runsCompleted >= 1`.

Liveness:

- `RunningMakesProgress` — a non-stale `running` job eventually leaves.
- `FailureIsRetried` — a `failed` job with budget left eventually moves on.
- `OneShotTerminates` — a one-shot job that starts eventually terminates.
- `RemovedAbsorbs` — `removed` is absorbing.

**AgentLoop**

- `TypeOK`, `ValidState`.
- `MessagesNonEmptyOnCall` (INV-A3) — `CALLING_API` implies the message
  list is non-empty.
- `BudgetMonotonic` (INV-A4) — the iteration counter never exceeds
  `IterationBudget + 1`.
- `RetryBounded` + `RetryResetsOnResponse` (INV-A5).
- `ApiParamsReadyOnCall` (INV-A6) — `CALLING_API` only after `BuildRequest`
  has run.
- `SystemPromptPreserved` (INV-A8) — compression and continuation never
  flip this flag.

Liveness:

- `Termination` — every execution eventually reaches
  `RETURNING_RESPONSE | INTERRUPTED | BUDGET_EXHAUSTED`.
- `InterruptLiveness` — once the interrupt flag is asserted, the machine
  eventually enters `INTERRUPTED`.
- `BudgetEventuallyFires` — running out of budget eventually terminates.
- `TerminalStaysTerminal` — absorbing terminal states.

## Mapping back to Python

`specs/REFINEMENT.md` gives the explicit simulation relation between the
TLA+ actions here and the Python classes.  Summary table:

| TLA+ action in AgentLoop.tla | Python method                              |
|------------------------------|--------------------------------------------|
| `ReceiveMessage`             | `AgentLoopStateMachine.receive_message`    |
| `BuildRequest`               | `.build_request`                           |
| `ReceiveResponse`            | `.receive_response`                        |
| `RecoverError`               | `.recover_error`                           |
| `DispatchTools`              | `.dispatch_tools`                          |
| `ProcessTextResponse`        | `.process_text_response`                   |
| `ToolComplete`               | `.tool_complete`                           |
| `Compress`                   | `.compress`                                |
| `ContinueGeneration`         | `.continue_generation`                     |
| `ContinueAfterCompression`   | `.continue_after_compression`              |
| `ReturnResultTurnDone`       | `.return_result`  (guard: `turn_done`)     |
| `ReturnResultRetriesExhausted` | `.return_result` (guard: `retry_exhausted`) |
| `RetryAPI`                   | `.retry_api`                               |
| `Interrupt`                  | `.interrupt`                               |
| `ExhaustBudget`              | `.exhaust_budget`                          |

| TLA+ action in CronJob.tla | Python method                              |
|----------------------------|--------------------------------------------|
| `Create`                   | `CronJobStateMachine.create`               |
| `Tick`                     | `.tick`                                    |
| `MarkSuccessRecurring`     | `.mark_success`  (recurring branch)        |
| `MarkSuccessDone`          | `.mark_success`  (one-shot / exhausted)    |
| `MarkFailure`              | `.mark_failure`                            |
| `Retry`                    | `.retry`                                   |
| `Pause`                    | `.pause`                                   |
| `Resume`                   | `.resume`                                  |
| `Trigger`                  | `.trigger`                                 |
| `Remove`                   | `.remove`                                  |
| `Recover`                  | `.recover`                                 |
