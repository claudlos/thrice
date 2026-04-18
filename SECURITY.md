# Security Policy

## Supported versions

Thrice is distributed as a rolling release off `main`. Only the latest
commit on `main` receives security fixes.

## Reporting a vulnerability

If you believe you've found a security issue, **do not open a public
issue**.  Instead:

1. Go to <https://github.com/claudlos/thrice/security/advisories/new>
   and file a private advisory, OR
2. Open an issue with the single word "security" in the title and no
   other details, and a maintainer will follow up by email.

Please include, at minimum:

- A short description of the issue and its impact.
- Steps or a proof-of-concept that triggers it.
- The commit hash or tag you observed it on.

You will normally get an initial reply within **7 days**.

## In scope

- The Python modules in `modules/`.
- The installer / updater / uninstaller scripts.
- Any code path that handles untrusted inputs (user prompts, tool
  outputs, file paths, shell arguments).

## Out of scope

- Hermes Agent itself — please report those upstream at
  <https://github.com/NousResearch/hermes-agent>.
- LLM jailbreaks that do not involve a Thrice-specific weakness.
- Issues only reproducible with Python <3.10.

## Known defensive hardening

The audit-driven hardening already in `main`:

- `context_mentions._resolve_file` constrains `@file:` paths to the
  working directory (no `..`, no absolute paths, no symlink escapes).
- Every `subprocess.run` call uses `List[str]` argv and a `timeout`.
- No `shell=True`, no `eval`, no `exec`, no `pickle.loads` anywhere in
  the shipped modules.
- The installer validates backup paths before `rmtree`.
