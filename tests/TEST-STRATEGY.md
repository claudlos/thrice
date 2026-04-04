# Test Strategy for Hermes Improvements

## Testing Pyramid

```
         /\
        /  \        Refinement Tests (abstract vs concrete)
       /    \       Catches: spec drift, simulation relation breaks
      /------\
     /        \     Property Tests (hypothesis stateful + @given)
    /          \    Catches: invariant violations under random ops,
   /            \   edge cases humans miss, concurrency-style bugs
  /--------------\
 /                \ Invariant Unit Tests (deterministic)
/                  \ Catches: each checker method works correctly
/--------------------\
|   Existing Tests    | Standard unit + integration tests
+---------------------+ Catches: functional regressions
```

## Test Levels

### Level 1: Invariant Unit Tests (`tests/invariant_tests/`)

**Purpose:** Verify each invariant checker method in isolation.

**What it catches:**
- Incorrect detection of valid states (false positives)
- Missed detection of invalid states (false negatives)
- Edge cases: empty lists, None fields, boundary values

**Files:**
- `test_runtime_invariants.py` — Tests for `hermes_invariants.InvariantChecker`
  (the new-files/hermes_invariants.py module). Covers message integrity, cron
  invariants, process registry, session invariants, delegation checks.
  Also tests enforcement modes (strict/warn/silent).
- `test_cron_invariants.py` — Tests for every invariant from SM-1 spec
  (INV-CJ1 through INV-CJ9), state transitions, state inference, liveness
  warnings, and the `validate_jobs_list()` integration function.

### Level 2: Property-Based Tests (`tests/property_tests/`)

**Purpose:** Randomly explore state spaces to find invariant violations.

**What it catches:**
- Invariant violations under sequences of operations humans wouldn't try
- State machine bugs only reachable through specific operation sequences
- Conservation law violations (budget, repeat counts)
- Concurrency-like issues (mutual exclusion)

**Files:**
- `test_cron_properties.py` — RuleBasedStateMachine for SM-1 cron lifecycle.
  Random create/tick/pause/resume/remove/mark_success/mark_failure operations
  with 10 invariant checks after every step. Also standalone @given tests for
  transition properties.
- `test_message_integrity.py` — Property tests for message list well-formedness.
  Strategies generate both well-formed and malformed conversations. Properties
  verify role validation, tool_call/result pairing, ordering, first message, etc.
- `test_process_lifecycle.py` — RuleBasedStateMachine for SM-4 process lifecycle.
  Random spawn/kill/poll/prune/finish with 7 invariant checks. Also standalone
  @given tests for transition properties and registry consistency.
- `test_delegation_tree.py` — RuleBasedStateMachine for SM-3 delegation tree.
  Random delegate_single/delegate_batch/child_works/child_completes/child_fails
  with 8 invariant checks including depth bound, budget conservation, acyclicity.

### Level 3: Refinement Tests (`tests/refinement_tests/`)

**Purpose:** Verify concrete implementation refines abstract specification.

**What it catches:**
- Spec drift: implementation transitions not allowed by spec
- Missing mappings: concrete states with no abstract equivalent
- Invariant preservation: abstract invariants broken in concrete
- Operation simulation: abstract operations map correctly to concrete ops

**Files:**
- `test_cron_refinement.py` — Checks simulation relation between SM-1 abstract
  state machine and `cron_invariants.py` concrete implementation. 6 test classes
  covering totality, mapping, transition refinement, invariant preservation,
  operation simulation, and table completeness.


## Run Instructions

### Run all tests
```bash
cd /path/to/hermes-improvements
python -m pytest tests/ -v
```

### Run by level
```bash
# Invariant unit tests only
python -m pytest tests/invariant_tests/ -v -m invariant

# Property-based tests only
python -m pytest tests/property_tests/ -v -m property

# Refinement tests only
python -m pytest tests/refinement_tests/ -v -m refinement

# Stateful (slow) tests only
python -m pytest tests/property_tests/ -v -m stateful
```

### Hypothesis profiles
```bash
# CI (fast, 30 examples, no shrinking)
HYPOTHESIS_PROFILE=ci python -m pytest tests/property_tests/ -v

# Dev (moderate, 100 examples)
HYPOTHESIS_PROFILE=dev python -m pytest tests/property_tests/ -v

# Exhaustive (nightly, 500 examples, verbose)
HYPOTHESIS_PROFILE=exhaustive python -m pytest tests/property_tests/ -v
```

### Run existing thrice tests alongside
```bash
# Thrice invariant tests
python -m pytest thrice/tests/ -v

# Everything
python -m pytest tests/ thrice/tests/ -v
```


## Hypothesis Settings

| Profile    | max_examples | stateful_step_count | shrinking | deadline |
|------------|-------------|---------------------|-----------|----------|
| ci         | 30          | 15                  | no        | none     |
| dev        | 100         | 30                  | yes       | none     |
| exhaustive | 500         | 50                  | yes       | none     |

Settings are configured in `tests/conftest.py` and can be overridden via
the `HYPOTHESIS_PROFILE` environment variable.


## State Machines Tested

| SM   | Name              | States | Transitions | Invariants Checked |
|------|-------------------|--------|-------------|-------------------|
| SM-1 | Cron Lifecycle    | 7      | 13          | 10                |
| SM-3 | Delegation Tree   | 6+5    | 16          | 8                 |
| SM-4 | Process Lifecycle  | 6      | 10          | 7                 |
| —    | Message List      | n/a    | n/a         | 7 (INV-M1..M7)   |


## Coverage Summary

| Invariant ID | Spec         | Unit Test | Property Test | Refinement |
|-------------|-------------|-----------|---------------|------------|
| INV-CJ1     | SM-1        | ✓         | ✓ (stateful)  | ✓          |
| INV-CJ2     | SM-1        | ✓         | ✓ (stateful)  | —          |
| INV-CJ3     | SM-1        | ✓         | —             | —          |
| INV-CJ4     | SM-1        | ✓         | —             | —          |
| INV-CJ5     | SM-1        | ✓         | —             | —          |
| INV-CJ6     | SM-1        | ✓         | ✓ (stateful)  | —          |
| INV-CJ7     | SM-1        | ✓         | —             | —          |
| INV-CJ8     | SM-1        | ✓         | —             | —          |
| INV-CJ9     | SM-1        | ✓         | ✓ (stateful)  | —          |
| INV-C1      | SM-1 spec   | ✓         | ✓ (stateful)  | ✓          |
| INV-C2      | SM-1 spec   | ✓         | ✓ (stateful)  | ✓          |
| INV-C3      | SM-1 spec   | —         | ✓ (stateful)  | —          |
| INV-C5      | SM-1 spec   | —         | ✓ (stateful)  | —          |
| INV-C6      | SM-1 spec   | —         | ✓ (stateful)  | —          |
| INV-C7      | SM-1 spec   | —         | ✓ (stateful)  | —          |
| INV-M1      | messages    | ✓         | ✓ (@given)    | —          |
| INV-M2      | messages    | ✓         | ✓ (@given)    | —          |
| INV-M3      | messages    | ✓         | ✓ (@given)    | —          |
| INV-M4      | messages    | ✓         | ✓ (@given)    | —          |
| INV-M5      | messages    | ✓         | ✓ (@given)    | —          |
| INV-M6      | messages    | ✓         | ✓ (@given)    | —          |
| INV-M7      | messages    | ✓         | ✓ (@given)    | —          |
| INV-P1      | SM-4        | ✓         | ✓ (stateful)  | —          |
| INV-P2      | SM-4        | ✓         | ✓ (stateful)  | —          |
| INV-P3      | SM-4        | ✓         | ✓ (stateful)  | —          |
| INV-P4      | SM-4        | ✓         | ✓ (stateful)  | —          |
| INV-P5      | SM-4        | ✓         | ✓ (stateful)  | —          |
| INV-P6      | SM-4        | ✓         | ✓ (stateful)  | —          |
| INV-P9      | SM-4        | ✓         | ✓ (stateful)  | —          |
| INV-D1      | SM-3        | ✓         | ✓ (stateful)  | —          |
| INV-D2      | SM-3        | ✓         | ✓ (stateful)  | —          |
| INV-D5      | SM-3        | ✓         | ✓ (stateful)  | —          |
| INV-D7      | SM-3        | ✓         | ✓ (stateful)  | —          |
