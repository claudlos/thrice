<!--
Thanks for the PR!  A short description + the checklist below is all we need.
-->

## Summary

<!-- What changes and why.  Link the audit / issue item if applicable. -->

## Checklist

- [ ] `pytest -q` is green locally.
- [ ] If a state machine changed, `./specs/tla/run_tlc.sh` still passes and `specs/REFINEMENT.md` is updated.
- [ ] If a new module was added, it is listed in `install.py::STANDALONE_MODULES`.
- [ ] `install.py --dry-run` still completes against a stock hermes-agent checkout.
- [ ] `CHANGELOG.md` has an entry under `## [Unreleased]`.
- [ ] No new `shell=True`, `eval`, `exec`, or `pickle.loads` on untrusted data.
