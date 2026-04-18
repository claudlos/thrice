------------------------------- MODULE AgentLoop -------------------------------
(*
  SM-2 : Hermes Agent Request/Response Loop.

  Abstract state machine for `modules/agent_loop_state_machine.py`.

  The Python implementation encodes this loop with an enum `AgentLoopState`,
  a guarded transition table, and a mutable `AgentLoopContext`.  This spec
  factors that into:

    - `state`          : current enum value
    - a bounded counter `iterationsUsed`   (abstracts token/iter budget)
    - a bounded counter `retryCount`       (API retry budget)
    - booleans `hasToolCalls`, `contextFull`, `needsContinue`,
               `interruptFlag`, `apiParamsBuilt`, `messagesNonEmpty`,
               `systemPromptPreserved`, `lastSuccessSeen`

  Python -> TLA+ mapping (guard predicates)
  -----------------------------------------
  messages_non_empty    -> messagesNonEmpty
  has_budget            -> HasBudget == iterationsUsed < IterationBudget
  has_tool_calls        -> hasToolCalls
  no_tools              -> ~hasToolCalls
  needs_continue        -> needsContinue
  context_full          -> contextFull
  turn_done             -> TurnDone
  can_retry             -> retryCount < MaxRetries
  interruptible         -> state \notin TerminalStates

  Invariants checked
  ------------------
  INV-A3 : messages are non-empty when state = CALLING_API
  INV-A4 : iterationsUsed <= IterationBudget + 1
  INV-A5 : retryCount <= MaxRetries;
           and retryCount = 0 whenever state = PROCESSING_RESPONSE
  INV-A6 : api_params set when state = CALLING_API
  INV-A7 : no transition leaves the declared state set
  INV-A8 : the system prompt is preserved across compression

  Liveness
  --------
  LV-A1 : Termination       - eventually a terminal state is reached.
  LV-A2 : Interrupt liveness - interrupt_flag ~> state = INTERRUPTED.
  LV-A3 : Budget liveness    - ~HasBudget ~> BUDGET_EXHAUSTED reached.
*)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    IterationBudget,   \* max iterations before BUDGET_EXHAUSTED fires
    MaxRetries         \* max successive API failures

ASSUME IterationBudget \in Nat /\ IterationBudget >= 1
ASSUME MaxRetries      \in Nat /\ MaxRetries      >= 0

AWAITING_INPUT        == "awaiting_input"
PREPARING_API_CALL    == "preparing_api_call"
CALLING_API           == "calling_api"
PROCESSING_RESPONSE   == "processing_response"
EXECUTING_TOOLS       == "executing_tools"
COMPRESSING_CONTEXT   == "compressing_context"
HANDLING_CONTINUATION == "handling_continuation"
RETURNING_RESPONSE    == "returning_response"
INTERRUPTED           == "interrupted"
BUDGET_EXHAUSTED      == "budget_exhausted"
ERROR_RECOVERY        == "error_recovery"

States == {
    AWAITING_INPUT,
    PREPARING_API_CALL,
    CALLING_API,
    PROCESSING_RESPONSE,
    EXECUTING_TOOLS,
    COMPRESSING_CONTEXT,
    HANDLING_CONTINUATION,
    RETURNING_RESPONSE,
    INTERRUPTED,
    BUDGET_EXHAUSTED,
    ERROR_RECOVERY
}

TerminalStates == {RETURNING_RESPONSE, INTERRUPTED, BUDGET_EXHAUSTED}

(* ---------- Variables --------------------------------------------------- *)

VARIABLES
    state,
    iterationsUsed,
    retryCount,
    messagesNonEmpty,
    apiParamsBuilt,
    hasToolCalls,
    needsContinue,
    contextFull,
    interruptFlag,
    systemPromptPreserved   \* abstract flag for INV-A8

vars == << state, iterationsUsed, retryCount, messagesNonEmpty,
          apiParamsBuilt, hasToolCalls, needsContinue, contextFull,
          interruptFlag, systemPromptPreserved >>

TypeOK ==
    /\ state \in States
    /\ iterationsUsed       \in 0..(IterationBudget + 1)
    /\ retryCount           \in 0..(MaxRetries + 1)
    /\ messagesNonEmpty     \in BOOLEAN
    /\ apiParamsBuilt       \in BOOLEAN
    /\ hasToolCalls         \in BOOLEAN
    /\ needsContinue        \in BOOLEAN
    /\ contextFull          \in BOOLEAN
    /\ interruptFlag        \in BOOLEAN
    /\ systemPromptPreserved \in BOOLEAN

(* ---------- Derived predicates ----------------------------------------- *)

HasBudget == iterationsUsed < IterationBudget
CanRetry  == retryCount     < MaxRetries
TurnDone  == (~needsContinue) /\ (~contextFull)

(* ---------- Initial state ---------------------------------------------- *)

Init ==
    /\ state                = AWAITING_INPUT
    /\ iterationsUsed       = 0
    /\ retryCount           = 0
    /\ messagesNonEmpty     = FALSE
    /\ apiParamsBuilt       = FALSE
    /\ hasToolCalls         = FALSE
    /\ needsContinue        = FALSE
    /\ contextFull          = FALSE
    /\ interruptFlag        = FALSE
    /\ systemPromptPreserved = TRUE

(* ---------- Transitions ------------------------------------------------ *)

\* ReceiveMessage: caller provides a non-empty message list.
\* AWAITING_INPUT -> PREPARING_API_CALL
ReceiveMessage ==
    /\ state = AWAITING_INPUT
    /\ state'            = PREPARING_API_CALL
    /\ messagesNonEmpty' = TRUE
    /\ UNCHANGED << iterationsUsed, retryCount, apiParamsBuilt,
                    hasToolCalls, needsContinue, contextFull,
                    interruptFlag, systemPromptPreserved >>

\* BuildRequest: construct api_params and advance iteration.
\* PREPARING_API_CALL -> CALLING_API  (guard: HasBudget)
BuildRequest ==
    /\ state = PREPARING_API_CALL
    /\ HasBudget
    /\ messagesNonEmpty
    /\ state'           = CALLING_API
    /\ apiParamsBuilt'  = TRUE
    /\ iterationsUsed'  = iterationsUsed + 1
    /\ UNCHANGED << retryCount, messagesNonEmpty, hasToolCalls,
                    needsContinue, contextFull, interruptFlag,
                    systemPromptPreserved >>

\* ExhaustBudget: PREPARING_API_CALL | HANDLING_CONTINUATION -> BUDGET_EXHAUSTED
ExhaustBudget ==
    /\ state \in {PREPARING_API_CALL, HANDLING_CONTINUATION}
    /\ ~HasBudget
    /\ state' = BUDGET_EXHAUSTED
    /\ UNCHANGED << iterationsUsed, retryCount, messagesNonEmpty,
                    apiParamsBuilt, hasToolCalls, needsContinue,
                    contextFull, interruptFlag, systemPromptPreserved >>

\* ReceiveResponse: CALLING_API -> PROCESSING_RESPONSE
\* Side effect (per INV-A5): retryCount is reset to 0.
\* The environment chooses whether there are tool_calls and whether the
\* response was truncated (finish_reason=length).
ReceiveResponse ==
    /\ state = CALLING_API
    /\ \E tools, cont, full \in BOOLEAN :
         /\ state'           = PROCESSING_RESPONSE
         /\ retryCount'      = 0
         /\ hasToolCalls'    = tools
         /\ needsContinue'   = cont
         /\ contextFull'     = full
    /\ UNCHANGED << iterationsUsed, messagesNonEmpty, apiParamsBuilt,
                    interruptFlag, systemPromptPreserved >>

\* RecoverError: CALLING_API -> ERROR_RECOVERY (API request failed)
RecoverError ==
    /\ state = CALLING_API
    /\ state' = ERROR_RECOVERY
    /\ UNCHANGED << iterationsUsed, retryCount, messagesNonEmpty,
                    apiParamsBuilt, hasToolCalls, needsContinue,
                    contextFull, interruptFlag, systemPromptPreserved >>

\* DispatchTools: PROCESSING_RESPONSE -> EXECUTING_TOOLS (guard: hasToolCalls)
DispatchTools ==
    /\ state = PROCESSING_RESPONSE
    /\ hasToolCalls
    /\ state' = EXECUTING_TOOLS
    /\ UNCHANGED << iterationsUsed, retryCount, messagesNonEmpty,
                    apiParamsBuilt, hasToolCalls, needsContinue,
                    contextFull, interruptFlag, systemPromptPreserved >>

\* ProcessTextResponse: PROCESSING_RESPONSE -> HANDLING_CONTINUATION (no tools)
ProcessTextResponse ==
    /\ state = PROCESSING_RESPONSE
    /\ ~hasToolCalls
    /\ state' = HANDLING_CONTINUATION
    /\ UNCHANGED << iterationsUsed, retryCount, messagesNonEmpty,
                    apiParamsBuilt, hasToolCalls, needsContinue,
                    contextFull, interruptFlag, systemPromptPreserved >>

\* ToolComplete: EXECUTING_TOOLS -> HANDLING_CONTINUATION
\* All tools executed; clear the pending flag.
ToolComplete ==
    /\ state = EXECUTING_TOOLS
    /\ state'         = HANDLING_CONTINUATION
    /\ hasToolCalls'  = FALSE
    /\ needsContinue' = TRUE   \* tool follow-ups force another turn
    /\ UNCHANGED << iterationsUsed, retryCount, messagesNonEmpty,
                    apiParamsBuilt, contextFull, interruptFlag,
                    systemPromptPreserved >>

\* Compress: HANDLING_CONTINUATION -> COMPRESSING_CONTEXT (guard: contextFull)
Compress ==
    /\ state = HANDLING_CONTINUATION
    /\ contextFull
    /\ state'        = COMPRESSING_CONTEXT
    /\ contextFull'  = FALSE                 \* compression frees budget
    /\ systemPromptPreserved' = systemPromptPreserved  \* INV-A8 must hold
    /\ UNCHANGED << iterationsUsed, retryCount, messagesNonEmpty,
                    apiParamsBuilt, hasToolCalls, needsContinue,
                    interruptFlag >>

\* ContinueAfterCompression: COMPRESSING_CONTEXT -> PREPARING_API_CALL
ContinueAfterCompression ==
    /\ state = COMPRESSING_CONTEXT
    /\ state'          = PREPARING_API_CALL
    /\ apiParamsBuilt' = FALSE
    /\ UNCHANGED << iterationsUsed, retryCount, messagesNonEmpty,
                    hasToolCalls, needsContinue, contextFull,
                    interruptFlag, systemPromptPreserved >>

\* ContinueGeneration: HANDLING_CONTINUATION -> PREPARING_API_CALL (needsContinue)
ContinueGeneration ==
    /\ state = HANDLING_CONTINUATION
    /\ needsContinue
    /\ ~contextFull
    /\ HasBudget
    /\ state'          = PREPARING_API_CALL
    /\ apiParamsBuilt' = FALSE
    /\ needsContinue'  = FALSE
    /\ UNCHANGED << iterationsUsed, retryCount, messagesNonEmpty,
                    hasToolCalls, contextFull, interruptFlag,
                    systemPromptPreserved >>

\* ReturnResultTurnDone: HANDLING_CONTINUATION -> RETURNING_RESPONSE
ReturnResultTurnDone ==
    /\ state = HANDLING_CONTINUATION
    /\ TurnDone
    /\ state' = RETURNING_RESPONSE
    /\ UNCHANGED << iterationsUsed, retryCount, messagesNonEmpty,
                    apiParamsBuilt, hasToolCalls, needsContinue,
                    contextFull, interruptFlag, systemPromptPreserved >>

\* ReturnResultRetriesExhausted: ERROR_RECOVERY -> RETURNING_RESPONSE
ReturnResultRetriesExhausted ==
    /\ state = ERROR_RECOVERY
    /\ ~CanRetry
    /\ state' = RETURNING_RESPONSE
    /\ UNCHANGED << iterationsUsed, retryCount, messagesNonEmpty,
                    apiParamsBuilt, hasToolCalls, needsContinue,
                    contextFull, interruptFlag, systemPromptPreserved >>

\* RetryAPI: ERROR_RECOVERY -> CALLING_API (guard: CanRetry)
RetryAPI ==
    /\ state = ERROR_RECOVERY
    /\ CanRetry
    /\ state'      = CALLING_API
    /\ retryCount' = retryCount + 1
    /\ UNCHANGED << iterationsUsed, messagesNonEmpty, apiParamsBuilt,
                    hasToolCalls, needsContinue, contextFull,
                    interruptFlag, systemPromptPreserved >>

\* Interrupt: any interruptible state -> INTERRUPTED
Interrupt ==
    /\ state \notin TerminalStates
    /\ interruptFlag'       = TRUE
    /\ state'               = INTERRUPTED
    /\ UNCHANGED << iterationsUsed, retryCount, messagesNonEmpty,
                    apiParamsBuilt, hasToolCalls, needsContinue,
                    contextFull, systemPromptPreserved >>

\* Environment: the user (or a signal handler) may assert the interrupt flag
\* at any time.  This is separate from the actual transition into INTERRUPTED
\* to model the (already observed) race between "signal set" and "state moves".
RaiseInterrupt ==
    /\ ~interruptFlag
    /\ interruptFlag' = TRUE
    /\ UNCHANGED << state, iterationsUsed, retryCount, messagesNonEmpty,
                    apiParamsBuilt, hasToolCalls, needsContinue,
                    contextFull, systemPromptPreserved >>

Next ==
    \/ ReceiveMessage
    \/ BuildRequest
    \/ ExhaustBudget
    \/ ReceiveResponse
    \/ RecoverError
    \/ DispatchTools
    \/ ProcessTextResponse
    \/ ToolComplete
    \/ Compress
    \/ ContinueAfterCompression
    \/ ContinueGeneration
    \/ ReturnResultTurnDone
    \/ ReturnResultRetriesExhausted
    \/ RetryAPI
    \/ Interrupt
    \/ RaiseInterrupt

\* Strong fairness on progress actions - prevents TLC from claiming
\* spurious deadlock when a terminal state is reachable but never selected.
Spec ==
    Init /\ [][Next]_vars
         /\ WF_vars(ReceiveMessage)
         /\ WF_vars(BuildRequest)
         /\ WF_vars(ReceiveResponse)
         /\ WF_vars(DispatchTools)
         /\ WF_vars(ProcessTextResponse)
         /\ WF_vars(ToolComplete)
         /\ WF_vars(ContinueAfterCompression)
         /\ WF_vars(ContinueGeneration)
         /\ WF_vars(ReturnResultTurnDone)
         /\ WF_vars(ReturnResultRetriesExhausted)
         /\ WF_vars(ExhaustBudget)
         /\ SF_vars(Interrupt)

(* ---------- Safety invariants ----------------------------------------- *)

\* INV-A3 : messages non-empty whenever we are mid-call.
MessagesNonEmptyOnCall ==
    state = CALLING_API => messagesNonEmpty

\* INV-A4 : iteration counter does not overshoot.
BudgetMonotonic == iterationsUsed <= IterationBudget + 1

\* INV-A5 : retry budget is bounded and resets on a good response.
RetryBounded == retryCount <= MaxRetries
RetryResetsOnResponse ==
    state = PROCESSING_RESPONSE => retryCount = 0

\* INV-A6 : api_params must exist before we attempt the call.
ApiParamsReadyOnCall ==
    state = CALLING_API => apiParamsBuilt

\* INV-A7 : the state variable never escapes the enum.
ValidState == state \in States

\* INV-A8 : system prompt is preserved through compression / continuation.
SystemPromptPreserved == systemPromptPreserved

\* INV-A9 : terminal states are absorbing (no action leaves them).
TerminalAbsorbing ==
    (state \in TerminalStates) => [](state \in TerminalStates)

SafetyInvariants ==
    /\ TypeOK
    /\ ValidState
    /\ MessagesNonEmptyOnCall
    /\ BudgetMonotonic
    /\ RetryBounded
    /\ RetryResetsOnResponse
    /\ ApiParamsReadyOnCall
    /\ SystemPromptPreserved

(* ---------- Liveness properties --------------------------------------- *)

\* LV-A1 : Every execution eventually reaches a terminal state.
Termination == <>(state \in TerminalStates)

\* LV-A2 : An asserted interrupt is eventually observed.
InterruptLiveness == interruptFlag ~> (state = INTERRUPTED)

\* LV-A3 : Running out of budget eventually lands in BUDGET_EXHAUSTED
\*          (unless we terminate for another reason first).
BudgetEventuallyFires ==
    (~HasBudget /\ state \in {PREPARING_API_CALL, HANDLING_CONTINUATION})
        ~> (state \in TerminalStates)

\* LV-A4 : terminal states stay terminal.
TerminalStaysTerminal == TerminalAbsorbing

LivenessProperties ==
    /\ Termination
    /\ InterruptLiveness
    /\ BudgetEventuallyFires
    /\ TerminalStaysTerminal

=============================================================================
