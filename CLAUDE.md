# Config-Driven AI Agent Framework

Hexagonal core; one engine, many agents defined purely by config. Guiding
principle: **ENGINE ≠ IDENTITY** — one engine consumes a typed, validated
config object and becomes different agents. No branching on agent name
anywhere.

Built incrementally in phases. Phase 0 delivered the configuration
foundation (models, secret references, loader, errors, profiles, tests).
Phase 1 added the tools layer: Tool interface, effect vocabulary, registry,
the pure PolicyGate, and the single enforcement chokepoint, plus three
built-in filesystem tools. Phase 2 (this codebase today) makes it run: the
LLMPort contract with an Anthropic adapter, a bounded reactive loop driving
tools only through the gate, a serializable session with enforced budgets
and a hard iteration ceiling, presence-aware confirmation with
suspend/resume, and a thin CLI. No network/web or shell tools, no
persistent memory or summarization, no planning, no events, voice, browser,
workflows, skills, MCP, or rich UI until their phases arrive.

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
8. **One enforcement chokepoint.** Tools are never executed except through
   the single enforcement wrapper, which consults the policy gate first.
   Tools contain zero permission logic; the gate is the only component that
   decides allow / deny / confirm. No code path may execute a tool while
   bypassing the gate.
9. **Confirmation means a human decides.** When the gate returns
   `REQUIRE_CONFIRMATION`: if a human is present, ask them; an explicit
   human "no" is the only thing that becomes a denial. If no human is
   available, **suspend** the run (record the pending action, await
   approval) — never auto-approve, never auto-deny, and never let the model
   route around a required confirmation by choosing a different tool.
10. **Every run is bounded.** No unbounded loops. Every run terminates by
    completion, by a budget cap, or by a hard iteration ceiling. Budgets
    and the ceiling always apply; they are not configurable away.

## Dev notes

- Python 3.11+, Pydantic v2, PyYAML, pytest. Local venv at `.venv`.
- Run tests: `.venv\Scripts\python.exe -m pytest`
- Agent profiles live in `profiles/agents/{name}.yaml`; system configs in
  `profiles/system/{env}.yaml` (selected via `APP_ENV`, default `dev`);
  user configs in `profiles/users/`. Profiles load by NAME only
  (`^[a-z0-9][a-z0-9_-]*$`), and the resolved path is confined to its base
  directory via `core/paths.py` — the same helper the gate uses for
  sandbox containment.
- Secrets: the config bundle holds only provider → env-var **name**
  references (`Secrets`); presence is validated at load, and
  `Secrets.resolve_secret(provider)` reads the value from the environment
  on demand at the point of use. The guarantee is structural — no dump of
  the bundle can contain a key, because the value is never stored.
- Layout: `config/` (schemas, loader, secret references), `core/` (effects,
  Tool contracts in `core/base.py`, PolicyGate, `enforce_and_run`, audit,
  path confinement, `core/llm.py` LLMPort contract, `core/session.py`,
  `core/loop.py`), `tools/` (registry, built-in filesystem tools), `llm/`
  (provider adapters), `testing/` (FakeLLM double), `ui/` (thin CLI).
  Dependencies point inward: `ui` → `llm`/`tools` → `core` →
  `config.models`; `core` never imports `tools` or `llm`;
  `core/__init__.py` stays import-free to keep the graph acyclic.
- The loop's iteration ceiling (`core/loop.py: ITERATION_CEILING`) is a
  constant on purpose — Rule 10 forbids configuring it away. The session's
  conversation buffer is naive (no summarization); real context management
  arrives with the memory phase. Cost budgeting activates only when a
  provider has `pricing` configured; otherwise the cost cap logs as
  inactive and token/tool-call caps still apply.
- Run the CLI from the repo root: `python -m ui.cli scribe` (sandbox
  `fs_root: workspace` in `profiles/system/dev.yaml` resolves against the
  working directory).
