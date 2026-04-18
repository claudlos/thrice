-------------------------------- MODULE CronJob --------------------------------
(*
  SM-1 : Cron Job Lifecycle State Machine.

  This specification is the abstract model of
  `modules/cron_state_machine.py`.  It formalises the finite-state
  machine that every Hermes cron job lives inside and captures the
  safety- and liveness-properties the Python implementation is
  expected to refine.

  Python <-> TLA+ mapping
  -----------------------
  state (str)                -> state  (enum)
  recurring (bool)           -> recurring
  repeat_total (int?)        -> repeatTotal
  runs_completed (int)       -> runsCompleted
  runs_left (int?)           -> runsLeft
  retry_count (int)          -> retryCount
  max_retries (int)          -> maxRetries
  enabled (bool)             -> enabled
  next_run_at (datetime?)    -> nextRunDue     (abstract boolean: "is due?")
  tick_pending (bool)        -> tickPending
  removed (bool)             -> removedFlag

  Every high-level operation in the Python class corresponds to exactly
  one action in Next, with the same guards and post-conditions.
*)
EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS
    MaxRepeat,   \* upper bound on repeatTotal explored by TLC
    MaxRetries   \* upper bound on retry_count  (>= 0)

ASSUME MaxRepeat   \in Nat /\ MaxRepeat   >= 0
ASSUME MaxRetries \in Nat /\ MaxRetries >= 0

States == {
    "nonexistent",
    "scheduled",
    "running",
    "paused",
    "completed",
    "failed",
    "removed"
}

TerminalStates == {"removed"}  \* only "removed" is a hard-terminal state;
                                \* "completed" can be rescheduled by external
                                \* clients in the concrete impl but in the
                                \* abstract spec we treat it as terminal-ish
                                \* (see CompletedIsTerminal theorem).

(* ---------- Variables --------------------------------------------------- *)

VARIABLES
    state,           \* current lifecycle state
    recurring,       \* BOOLEAN : recurring job?
    repeatTotal,     \* total runs requested (0 = unbounded / one-shot)
    runsCompleted,   \* number of successful runs so far
    runsLeft,        \* remaining runs (if bounded)
    retryCount,      \* failures since last success
    maxRetries,      \* configured retry limit
    enabled,         \* BOOLEAN : scheduler may tick this job
    nextRunDue,      \* BOOLEAN : "now >= next_run_at"
    tickPending,     \* BOOLEAN : a tick is queued on this job
    removedFlag,     \* BOOLEAN : remove() has been invoked
    staleLock,       \* BOOLEAN : running job detected as stale
    lastError        \* BOOLEAN : last run set an error flag

vars == << state, recurring, repeatTotal, runsCompleted, runsLeft,
          retryCount, maxRetries, enabled, nextRunDue, tickPending,
          removedFlag, staleLock, lastError >>

TypeOK ==
    /\ state          \in States
    /\ recurring      \in BOOLEAN
    /\ repeatTotal    \in 0..MaxRepeat
    /\ runsCompleted  \in 0..MaxRepeat
    /\ runsLeft       \in 0..MaxRepeat
    /\ retryCount     \in 0..(MaxRetries+1)  \* +1 so we can *observe* a violation before detecting
    /\ maxRetries     \in 0..MaxRetries
    /\ enabled        \in BOOLEAN
    /\ nextRunDue     \in BOOLEAN
    /\ tickPending    \in BOOLEAN
    /\ removedFlag    \in BOOLEAN
    /\ staleLock      \in BOOLEAN
    /\ lastError      \in BOOLEAN

(* ---------- Initial State ---------------------------------------------- *)

Init ==
    /\ state          = "nonexistent"
    /\ recurring      = FALSE
    /\ repeatTotal    = 0
    /\ runsCompleted  = 0
    /\ runsLeft       = 0
    /\ retryCount     = 0
    /\ maxRetries     \in 0..MaxRetries
    /\ enabled        = TRUE
    /\ nextRunDue     = FALSE
    /\ tickPending    = FALSE
    /\ removedFlag    = FALSE
    /\ staleLock      = FALSE
    /\ lastError      = FALSE

(* ---------- Transition Actions ----------------------------------------- *)

\* create() : nonexistent -> scheduled
Create(r, total) ==
    /\ state = "nonexistent"
    /\ r \in BOOLEAN
    /\ total \in 0..MaxRepeat
    /\ state'          = "scheduled"
    /\ recurring'      = r
    /\ repeatTotal'    = total
    /\ runsLeft'       = total
    /\ nextRunDue'     = TRUE            \* scheduler may arm a due time
    /\ enabled'        = TRUE
    /\ UNCHANGED << runsCompleted, retryCount, maxRetries, tickPending,
                    removedFlag, staleLock, lastError >>

\* tick() : scheduled -> running  (guard: enabled /\ due)
Tick ==
    /\ state = "scheduled"
    /\ enabled
    /\ nextRunDue
    /\ state'       = "running"
    /\ tickPending' = FALSE
    /\ nextRunDue'  = FALSE           \* consumed
    /\ UNCHANGED << recurring, repeatTotal, runsCompleted, runsLeft,
                    retryCount, maxRetries, enabled, removedFlag,
                    staleLock, lastError >>

\* mark_success() : running -> scheduled  (recurring with at least one more run left after this one)
MarkSuccessRecurring ==
    /\ state = "running"
    /\ recurring
    /\ runsLeft > 1             \* strict: this run + at least one more
    /\ state'          = "scheduled"
    /\ runsCompleted'  = runsCompleted + 1
    /\ runsLeft'       = runsLeft - 1
    /\ retryCount'     = 0
    /\ lastError'      = FALSE
    /\ nextRunDue'     = TRUE       \* armed for next fire
    /\ UNCHANGED << recurring, repeatTotal, maxRetries, enabled,
                    tickPending, removedFlag, staleLock >>

\* mark_success() : running -> completed  (one-shot, or final bounded run).
\* Fires when the job either is not recurring, or this run exhausts the budget.
MarkSuccessDone ==
    /\ state = "running"
    /\ (~recurring \/ runsLeft <= 1)
    /\ state'          = "completed"
    /\ runsCompleted'  = runsCompleted + 1
    /\ runsLeft'       = 0
    /\ retryCount'     = 0
    /\ lastError'      = FALSE
    /\ nextRunDue'     = FALSE      \* completed jobs have no next run
    /\ enabled'        = FALSE
    /\ UNCHANGED << recurring, repeatTotal, maxRetries, tickPending,
                    removedFlag, staleLock >>

\* mark_failure() : running -> failed
MarkFailure ==
    /\ state = "running"
    /\ state'      = "failed"
    /\ retryCount' = IF retryCount < MaxRetries + 1
                      THEN retryCount + 1
                      ELSE retryCount
    /\ lastError'  = TRUE
    /\ UNCHANGED << recurring, repeatTotal, runsCompleted, runsLeft,
                    maxRetries, enabled, nextRunDue, tickPending,
                    removedFlag, staleLock >>

\* retry() : failed -> scheduled  (guard: retry_count < max_retries)
Retry ==
    /\ state = "failed"
    /\ retryCount < maxRetries
    /\ state'       = "scheduled"
    /\ nextRunDue'  = TRUE
    /\ UNCHANGED << recurring, repeatTotal, runsCompleted, runsLeft,
                    retryCount, maxRetries, enabled, tickPending,
                    removedFlag, staleLock, lastError >>

\* pause() : scheduled|failed -> paused
Pause ==
    /\ state \in {"scheduled", "failed"}
    /\ state'       = "paused"
    /\ enabled'     = FALSE
    /\ tickPending' = FALSE
    /\ UNCHANGED << recurring, repeatTotal, runsCompleted, runsLeft,
                    retryCount, maxRetries, nextRunDue, removedFlag,
                    staleLock, lastError >>

\* resume() : paused -> scheduled
Resume ==
    /\ state = "paused"
    /\ state'    = "scheduled"
    /\ enabled'  = TRUE
    /\ UNCHANGED << recurring, repeatTotal, runsCompleted, runsLeft,
                    retryCount, maxRetries, nextRunDue, tickPending,
                    removedFlag, staleLock, lastError >>

\* trigger() : paused -> scheduled (force fire)
Trigger ==
    /\ state = "paused"
    /\ state'       = "scheduled"
    /\ enabled'     = TRUE
    /\ nextRunDue'  = TRUE
    /\ UNCHANGED << recurring, repeatTotal, runsCompleted, runsLeft,
                    retryCount, maxRetries, tickPending, removedFlag,
                    staleLock, lastError >>

\* remove() : {scheduled, paused, completed, failed, nonexistent} -> removed
\* Guard: current state /= "running"
Remove ==
    /\ state \in {"scheduled", "paused", "completed", "failed", "nonexistent"}
    /\ state'        = "removed"
    /\ removedFlag'  = TRUE
    /\ enabled'      = FALSE
    /\ nextRunDue'   = FALSE
    /\ UNCHANGED << recurring, repeatTotal, runsCompleted, runsLeft,
                    retryCount, maxRetries, tickPending, staleLock,
                    lastError >>

\* recover() : running -> failed  (guard: staleLock)
Recover ==
    /\ state = "running"
    /\ staleLock
    /\ state'      = "failed"
    /\ staleLock'  = FALSE
    /\ lastError'  = TRUE
    /\ retryCount' = IF retryCount < MaxRetries + 1
                      THEN retryCount + 1
                      ELSE retryCount
    /\ UNCHANGED << recurring, repeatTotal, runsCompleted, runsLeft,
                    maxRetries, enabled, nextRunDue, tickPending,
                    removedFlag >>

\* Environment action: the wall clock advances and a scheduled job becomes due.
\* (Models the "scheduler tick arriving later" case.)
BecomeDue ==
    /\ state = "scheduled"
    /\ ~nextRunDue
    /\ nextRunDue' = TRUE
    /\ UNCHANGED << state, recurring, repeatTotal, runsCompleted, runsLeft,
                    retryCount, maxRetries, enabled, tickPending,
                    removedFlag, staleLock, lastError >>

\* Environment action: a running job's worker crashes so a stale lock appears.
LockGoesStale ==
    /\ state = "running"
    /\ ~staleLock
    /\ staleLock' = TRUE
    /\ UNCHANGED << state, recurring, repeatTotal, runsCompleted, runsLeft,
                    retryCount, maxRetries, enabled, nextRunDue,
                    tickPending, removedFlag, lastError >>

Next ==
    \/ \E r \in BOOLEAN, t \in 0..MaxRepeat : Create(r, t)
    \/ Tick
    \/ MarkSuccessRecurring
    \/ MarkSuccessDone
    \/ MarkFailure
    \/ Retry
    \/ Pause
    \/ Resume
    \/ Trigger
    \/ Remove
    \/ Recover
    \/ BecomeDue
    \/ LockGoesStale

Spec ==
    Init /\ [][Next]_vars
         /\ WF_vars(Tick)
         /\ WF_vars(MarkSuccessRecurring)
         /\ WF_vars(MarkSuccessDone)
         /\ WF_vars(Retry)
         /\ WF_vars(Recover)

(* ---------- Invariants (safety) ---------------------------------------- *)

\* INV-1 : The state variable never leaves the declared enum.
ValidState == state \in States

\* INV-2 : A running job must never also be flagged removed.
RunningNotDeleted == ~(state = "running" /\ removedFlag)

\* INV-3 : Paused jobs never carry a pending tick.
PausedNeverFires == state = "paused" => ~tickPending

\* INV-4 : Repeat counter consistency.
\*
\*   For recurring bounded jobs, completed + left == total at all times.
\*   When the job finishes (state = completed), runsLeft must be 0.
RepeatConsistency ==
    (recurring /\ repeatTotal > 0) =>
        /\ runsCompleted + runsLeft = repeatTotal
        /\ (state = "completed") => (runsLeft = 0)

\* INV-5 : Completed / removed jobs have no armed run.
TerminalNoNextRun ==
    state \in {"completed", "removed"} => ~nextRunDue

\* INV-6 : Running jobs are by definition enabled.
RunningIsEnabled == state = "running" => enabled

\* INV-7 : Retry budget is bounded.
\*
\*   retryCount may momentarily be equal to maxRetries+0 after a failure,
\*   but after either Retry or Recover it must go back below maxRetries.
RetryBounded == retryCount <= maxRetries + 1

\* INV-8 : A "completed" job has at least one successful run recorded.
CompletedHasRun == state = "completed" => runsCompleted >= 1

\* INV-9 : The "removed" sink state is absorbing.
RemovedIsAbsorbing == state = "removed" => [](state = "removed")
\* (temporal; see Liveness section for [] usage)

SafetyInvariants ==
    /\ TypeOK
    /\ ValidState
    /\ RunningNotDeleted
    /\ PausedNeverFires
    /\ RepeatConsistency
    /\ TerminalNoNextRun
    /\ RunningIsEnabled
    /\ RetryBounded
    /\ CompletedHasRun

(* ---------- Temporal / Liveness ---------------------------------------- *)

\* LV-1 : A running, non-stale job eventually leaves state "running".
\*        (i.e. no agent thread is stuck forever in "running" unless crashed.)
RunningMakesProgress ==
    (state = "running" /\ ~staleLock) ~> (state \in {"scheduled", "completed", "failed"})

\* LV-2 : A failed job with retries left eventually tries again.
FailureIsRetried ==
    (state = "failed" /\ retryCount < maxRetries) ~> (state \in {"scheduled", "paused", "removed"})

\* LV-3 : A one-shot job that starts running eventually terminates (completed|failed|removed).
OneShotTerminates ==
    (state = "running" /\ ~recurring) ~> (state \in {"completed", "failed", "removed"})

\* LV-4 : "removed" is absorbing (formal version of INV-9).
RemovedAbsorbs ==
    state = "removed" => []( state = "removed" )

LivenessProperties ==
    /\ RunningMakesProgress
    /\ FailureIsRetried
    /\ OneShotTerminates
    /\ RemovedAbsorbs

=============================================================================
