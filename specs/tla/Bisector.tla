------------------------------ MODULE Bisector ------------------------------
(*
  SM-3 : Regression Bisector.

  Abstract model of ``modules/regression_bisector.py``.  Formalises the
  state graph and the convergence property: given a non-empty interval
  ``[good, bad]`` where the regression is somewhere strictly between
  ``good`` and ``bad``, a sequence of classifications either narrows the
  interval (halving it each step, since TLC will pick any valid
  classification) to a single commit or reports an abort.

  State
  -----
    idle       -> no bisect running
    testing    -> the midpoint is being evaluated by the test command
    narrowing  -> git bisect has consumed the result and picked a new midpoint
    found      -> first bad commit identified (abstract: interval width = 1)
    completed  -> git bisect reset (terminal, absorbing)
    aborted    -> fatal error or timeout (terminal, absorbing)

  Variables
  ---------
    state        : BisectState
    lo, hi       : integers representing the current ``[good, bad]``
                   index range on the commit chain.  The regression lives
                   at integer ``target`` with ``lo < target <= hi``.
    target       : fixed across the run; only the bisect's view shrinks.
    steps        : number of test runs so far.
    aborted_bool : latched once; prevents reentry to non-terminal states.
*)
EXTENDS Naturals, TLC

CONSTANTS
    MaxCommits,       \* total commits in the range, 2..N
    MaxSteps          \* hard cap on bisect iterations (safety bound)

ASSUME MaxCommits \in Nat /\ MaxCommits >= 2
ASSUME MaxSteps   \in Nat /\ MaxSteps   >= 1

States == {"idle", "testing", "narrowing", "found", "completed", "aborted"}
TerminalStates == {"completed", "aborted"}

VARIABLES
    state,
    lo,
    hi,
    target,
    steps

vars == <<state, lo, hi, target, steps>>

TypeOK ==
    /\ state  \in States
    /\ lo     \in 0..MaxCommits
    /\ hi     \in 0..MaxCommits
    /\ target \in 0..MaxCommits
    /\ steps  \in 0..(MaxSteps + 1)

Init ==
    /\ state  = "idle"
    /\ lo     = 0
    /\ hi     \in 2..MaxCommits          \* at least one commit to narrow
    /\ target \in 1..hi                   \* strictly > lo, <= hi
    /\ steps  = 0

(* ---------- Transitions ------------------------------------------------ *)

\* Start: idle -> testing.  The bisector sets ``bad`` and ``good`` refs.
Start ==
    /\ state = "idle"
    /\ state'  = "testing"
    /\ UNCHANGED << lo, hi, target, steps >>

\* TestMidpoint: testing -> narrowing.  Environment picks the midpoint
\* and classifies it.  We model the classifier as knowledge of where
\* ``target`` lives: if target > mid, the commit is good (regression
\* came later); if target <= mid, the commit is bad (regression already
\* present).  The bisect proceeds by halving the interval.
Mid == (lo + hi) \div 2

TestMidpointGood ==
    /\ state = "testing"
    /\ Mid < target
    /\ state'  = "narrowing"
    /\ lo'     = Mid                      \* regression is strictly > Mid
    /\ steps'  = steps + 1
    /\ UNCHANGED << hi, target >>

TestMidpointBad ==
    /\ state = "testing"
    /\ Mid >= target
    /\ state'  = "narrowing"
    /\ hi'     = Mid                      \* regression is <= Mid
    /\ steps'  = steps + 1
    /\ UNCHANGED << lo, target >>

\* After each step the machine either has a single-commit interval
\* (-> found) or continues to pick the next midpoint (-> testing).
Converge ==
    /\ state = "narrowing"
    /\ hi - lo <= 1
    /\ state' = "found"
    /\ UNCHANGED << lo, hi, target, steps >>

Continue ==
    /\ state = "narrowing"
    /\ hi - lo > 1
    /\ state' = "testing"
    /\ UNCHANGED << lo, hi, target, steps >>

\* Found -> completed: the Python impl calls ``git bisect reset``.
Finish ==
    /\ state = "found"
    /\ state' = "completed"
    /\ UNCHANGED << lo, hi, target, steps >>

\* Abort: any non-terminal state -> aborted.  Models:
\*   - step timeout exceeded
\*   - total timeout exceeded
\*   - git error (e.g. bad ref)
Abort ==
    /\ state \notin TerminalStates
    /\ state' = "aborted"
    /\ UNCHANGED << lo, hi, target, steps >>

\* Safety: if MaxSteps is exceeded, the abstract model must abort (the
\* Python impl raises BisectError in exactly this case).
StepLimitAbort ==
    /\ state \in {"testing", "narrowing"}
    /\ steps >= MaxSteps
    /\ state' = "aborted"
    /\ UNCHANGED << lo, hi, target, steps >>

Next ==
    \/ Start
    \/ TestMidpointGood
    \/ TestMidpointBad
    \/ Converge
    \/ Continue
    \/ Finish
    \/ Abort
    \/ StepLimitAbort

Spec ==
    Init /\ [][Next]_vars
         /\ WF_vars(Start)
         /\ WF_vars(TestMidpointGood)
         /\ WF_vars(TestMidpointBad)
         /\ WF_vars(Converge)
         /\ WF_vars(Continue)
         /\ WF_vars(Finish)

(* ---------- Safety invariants ----------------------------------------- *)

\* B-INV-1 : the state variable never leaves the declared enum.
ValidState == state \in States

\* B-INV-2 : the tracking interval contains the target at all times,
\*           except in aborted/idle where correctness is not claimed.
IntervalContainsTarget ==
    state \in {"testing", "narrowing", "found", "completed"}
        => (lo < target /\ target <= hi)

\* B-INV-3 : the interval only shrinks (monotonic narrowing).
\*           Prime variables compared to current values on every step
\*           that leaves lo/hi unchanged or narrows them.
IntervalMonotonic == lo' >= lo /\ hi' <= hi

\* B-INV-4 : step count is bounded.
StepsBounded == steps <= MaxSteps + 1

\* B-INV-5 : ``found`` implies we've narrowed to a single candidate.
FoundIsSingleton ==
    state = "found" => (hi - lo <= 1)

\* B-INV-6 : terminal states are absorbing.
TerminalAbsorbing ==
    (state \in TerminalStates) => [](state \in TerminalStates)

SafetyInvariants ==
    /\ TypeOK
    /\ ValidState
    /\ IntervalContainsTarget
    /\ StepsBounded
    /\ FoundIsSingleton

(* ---------- Liveness -------------------------------------------------- *)

\* B-LV-1 : bisect always reaches a terminal state.
Termination == <>(state \in TerminalStates)

\* B-LV-2 : if we keep narrowing fairly and never abort, we eventually
\*           reach ``found`` on an arbitrary initial interval of width
\*           <= MaxSteps (the standard log-base-2 convergence).
ConvergesIfNoAbort ==
    [](state /= "aborted") => <>(state = "found" \/ state = "completed")

\* B-LV-3 : the terminal state is absorbing.
TerminalStaysTerminal == TerminalAbsorbing

LivenessProperties ==
    /\ Termination
    /\ ConvergesIfNoAbort
    /\ TerminalStaysTerminal

=============================================================================
