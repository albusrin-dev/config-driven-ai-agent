# Config-Driven AI Agent Framework

Hexagonal core; one engine, many agents defined purely by config. Guiding
principle: **ENGINE ≠ IDENTITY** — one engine consumes a typed, validated
config object and becomes different agents. No branching on agent name
anywhere.

Built incrementally in phases. Phase 0 (this codebase today) is the
configuration foundation only: models, secrets resolution, loader, errors,
example profiles, tests. No tools, memory, policy gate, run loop, LLM calls,
voice, browser, events, workflows, skills, or UI until their phases arrive.

## Durable Project Rules (all phases)

1. **Config is declarative data; logic is code.** No conditionals,
   expressions, loops, or business logic in YAML. The loader never `eval`s or
   executes anything from a config file.
2. **No agent-name branching.** No `if name == "JARVIS"` anywhere, ever. The
   engine and loader treat the name as an opaque identifier. This must be
   reviewable and true.
3. **One source of truth for capability.** What an agent can do is defined by
   its **tool allowlist**, not by duplicate capability booleans. Do not
   introduce parallel `filesystem: true`-style flags.
4. **Secrets come only from environment variables.** Never store a secret
   value in any config file. Config may reference the *name* of an env var;
   it may never contain the *value*.
5. **Safe, fail-closed defaults.** Unspecified permissions default to the
   most restrictive option. Unknown config fields are an error, not a
   warning.
6. **Validation is mandatory and loud.** Invalid config fails at load with a
   clear, specific message (which file, which field, what was wrong, what was
   allowed). No silent coercion of security-relevant fields.
7. **Extract interfaces on contact, not on speculation.** Do not add
   abstraction for phases not yet being built.

## Dev notes

- Python 3.11+, Pydantic v2, PyYAML, pytest. Local venv at `.venv`.
- Run tests: `.venv\Scripts\python.exe -m pytest`
- Agent profiles live in `profiles/agents/{name}.yaml`; system configs in
  `profiles/system/{env}.yaml` (selected via `APP_ENV`, default `dev`);
  user configs in `profiles/users/`.
- Secrets are resolved from the environment at load time into a separate
  in-memory `Secrets` object — never a field on a config model, never
  serialized.
