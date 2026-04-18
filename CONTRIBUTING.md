# Contributing to Thrice

Thrice is a focused add-on for Hermes Agent.  Contributions are welcome;
please read this short guide before opening a PR.

## Setup

```bash
git clone https://github.com/claudlos/thrice.git
cd thrice
python -m venv .venv && source .venv/bin/activate  # or Scripts\activate on Windows
pip install -e ".[dev]"
```

`pip install -e .` puts `modules/` on `sys.path` automatically, so tests
run with just:

```bash
pytest -q
```

Hermes-dependent tests skip cleanly when a live `hermes-agent` checkout
isn't on `PYTHONPATH`.

## Running the formal verification

```bash
# Requires Java 11+; the script auto-downloads tla2tools.jar
./specs/tla/run_tlc.sh
```

See [`specs/tla/README.md`](specs/tla/README.md) for what's checked and
[`specs/REFINEMENT.md`](specs/REFINEMENT.md) for the simulation relation
between the TLA+ specs and the Python state machines.

## What to change where

| Change                         | File(s)                                      |
|--------------------------------|----------------------------------------------|
| New standalone module          | `modules/<name>.py` + add to `STANDALONE_MODULES` in `install.py` |
| New patch to Hermes core       | `patches/<target>.patch` + add to `PATCH_MAP` |
| New state-machine invariant    | Python: `modules/<sm>.py` · TLA+: `specs/tla/<SM>.tla` + `MC<SM>.cfg` |
| New test                       | `tests/test_<module>.py` (mirrors module name) |

When you change a state machine, **change both sides**:
the Python module (and its Hypothesis tests) *and* the TLA+ spec (and
its TLC invariants).  The refinement suite in
`tests/refinement_tests/test_thrice_sm_refinement.py` will fail if they
drift.

## PR checklist

- [ ] `pytest -q` is green locally.
- [ ] If you touched a state machine, `./specs/tla/run_tlc.sh` still
      passes and `specs/REFINEMENT.md` is updated.
- [ ] If you added a new module, `install.py --dry-run` against a stock
      hermes-agent checkout still prints only `OK` lines.
- [ ] New code has tests, not just docs.
- [ ] No new `subprocess.run(..., shell=True)`, `eval`, `exec`, or
      `pickle.loads` on untrusted data.

## Code style

- Python 3.10+ typing (use `from __future__ import annotations` for
  newer syntax on older runtimes).
- `ruff check` in CI; see `pyproject.toml` for rules.  Line length is
  100, not 80.
- No `print()` — use `logger = logging.getLogger(__name__)`.
- Docstrings: one line required on every public class/function; the
  "why" is more valuable than the "what".
- No new module-level side effects (import must not open files, make
  network calls, or spawn subprocesses).

## Commit messages

Short imperative subject, wrapped at 72 chars.  If the change touches a
TLA+ invariant, name it:

```
fix(agent_loop): increment iterations_used after apply (INV-A4)

Guard previously saw post-increment value, causing BuildRequest
to fail when iteration_budget == 1.  Aligned with MCAgentLoop.cfg
spec action BuildRequest where iterationsUsed' update is atomic.
```

## License

By contributing you agree that your changes are released under the
project's MIT license (see [`LICENSE`](LICENSE)).
