"""Lightweight TLA+ coverage check.

For every named transition action, ensure every declared VARIABLE appears
either primed (X') or inside an UNCHANGED << ... >> clause.  This is a
heuristic - it does not replace TLC parsing, but catches the easy
"forgot to mark X' or UNCHANGED X" bug class.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path


SKIP_NAMES = {
    "Init", "TypeOK", "States", "TerminalStates", "vars", "HasBudget",
    "CanRetry", "TurnDone", "Next", "Spec", "ValidState", "TerminalAbsorbing",
    "SafetyInvariants", "LivenessProperties", "MessagesNonEmptyOnCall",
    "BudgetMonotonic", "RetryBounded", "RetryResetsOnResponse",
    "ApiParamsReadyOnCall", "SystemPromptPreserved", "Termination",
    "InterruptLiveness", "BudgetEventuallyFires", "TerminalStaysTerminal",
    "RunningNotDeleted", "PausedNeverFires", "RepeatConsistency",
    "TerminalNoNextRun", "RunningIsEnabled", "CompletedHasRun",
    "RemovedIsAbsorbing", "RunningMakesProgress", "FailureIsRetried",
    "OneShotTerminates", "RemovedAbsorbs",
    # Bisector.tla derived predicates & helpers
    "Mid", "IntervalContainsTarget", "IntervalMonotonic",
    "StepsBounded", "FoundIsSingleton", "ConvergesIfNoAbort",
}

COMMENT_RE = re.compile(r"\\\*.*")


def check(path: Path) -> int:
    src = path.read_text()
    block_match = re.search(r"VARIABLES\s+(.+?)(?=\n\n|\nTypeOK|\nvars)",
                            src, re.S)
    if not block_match:
        print(f"{path.name}: no VARIABLES block found")
        return 1
    block = COMMENT_RE.sub("", block_match.group(1))
    names = [v.strip().rstrip(",") for v in block.split(",")]
    names = [n for n in names if n]
    print(f"{path.name}: {len(names)} variables")

    action_re = re.compile(r"^(\w+) ==\s*\n(.+?)(?=^\w+ ==|\Z)", re.M | re.S)
    issues = 0
    actions_ok = 0
    for name, body in action_re.findall(src):
        if name in SKIP_NAMES:
            continue
        primed = set(re.findall(r"([A-Za-z]\w+)'", body))
        unchanged_match = re.search(r"UNCHANGED\s*<<\s*(.+?)\s*>>", body, re.S)
        unchanged: set[str] = set()
        if unchanged_match:
            raw = COMMENT_RE.sub("", unchanged_match.group(1))
            unchanged = {v.strip() for v in raw.split(",") if v.strip()}
        covered = primed | unchanged
        missing = [v for v in names if v not in covered]
        if missing:
            print(f"  {path.stem}.{name}: UNCOVERED vars: {missing}")
            issues += 1
        else:
            actions_ok += 1
    print(f"  {path.stem}: {actions_ok} clean transitions, {issues} with gaps")
    return issues


def main() -> int:
    here = Path(__file__).parent
    total = 0
    for spec in ("CronJob.tla", "AgentLoop.tla", "Bisector.tla"):
        total += check(here / spec)
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
