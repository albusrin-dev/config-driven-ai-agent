"""Thin CLI driver: read input -> run a turn -> print output -> prompt
inline for confirmations. A client of the core; holds no policy, no loop
logic, no provider knowledge beyond constructing the adapter.

Usage: python -m ui.cli <agent-name> [--env ENV]
"""

from __future__ import annotations

import argparse
from typing import Callable

from config.loader import LoadedConfig, load_agent, load_system_config
from core.loop import (
    BudgetExceeded,
    Completed,
    Errored,
    Suspended,
    resume,
    run_turn,
)
from core.session import new_session
from llm.anthropic import AnthropicAdapter
from tools.registry import ToolRegistry
from tools.builtins.files import DeleteFileTool, ReadFileTool, WriteFileTool

InputFn = Callable[[str], str]
PrintFn = Callable[[str], None]


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(DeleteFileTool())
    return registry


def make_cli_approver(input_fn: InputFn, print_fn: PrintFn):
    """Inline confirmation prompt: the human at the terminal decides
    (Durable Rule 9)."""

    def approver(decision) -> bool:
        print_fn(f"[confirm] {decision.reason}")
        for line in decision.effects_summary:
            print_fn(f"  effect: {line}")
        answer = input_fn("approve? [y/N] ")
        return answer.strip().lower() in ("y", "yes")

    return approver


def run_cli(
    loaded: LoadedConfig,
    llm,
    registry: ToolRegistry,
    input_fn: InputFn = input,
    print_fn: PrintFn = print,
) -> None:
    session = new_session(loaded.agent.name)
    approver = make_cli_approver(input_fn, print_fn)
    print_fn(f"agent '{loaded.agent.name}' ready (autonomy: "
             f"{loaded.agent.autonomy.value}). 'exit' or blank line to quit.")

    while True:
        try:
            line = input_fn("you> ")
        except EOFError:
            break
        if not line.strip() or line.strip().lower() == "exit":
            break

        result = run_turn(session, line, llm, registry,
                          loaded.agent, loaded.system, approver=approver)
        # With an inline approver, suspension is rare (e.g. piped stdin
        # closing); handle it anyway so the CLI can never dead-end.
        while isinstance(result, Suspended):
            answer = input_fn(
                f"[pending] {result.pending.reason} — approve? [y/N] "
            )
            result = resume(session, answer.strip().lower() in ("y", "yes"),
                            llm, registry, loaded.agent, loaded.system,
                            approver=approver)

        if isinstance(result, Completed):
            print_fn(result.text)
        elif isinstance(result, BudgetExceeded):
            print_fn(f"[stopped: '{result.which}' budget exhausted]")
            break
        elif isinstance(result, Errored):
            print_fn(f"[error] {result.reason}")
            break

    print_fn(
        f"[session {session.status}: tokens={session.budget.tokens_used}, "
        f"tool_calls={session.budget.tool_calls_made}, "
        f"cost=${session.budget.cost_used:.4f}]"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run an agent at the terminal.")
    parser.add_argument("agent", help="agent profile name (e.g. scribe)")
    parser.add_argument("--env", default=None,
                        help="environment name (default: APP_ENV or 'dev')")
    args = parser.parse_args(argv)

    system = load_system_config(args.env)
    loaded = load_agent(args.agent, system)
    provider = system.providers[loaded.agent.llm.provider]
    llm = AnthropicAdapter(
        provider_name=loaded.agent.llm.provider,
        model=loaded.agent.llm.model,
        secrets=loaded.secrets,
        temperature=loaded.agent.llm.temperature,
        system_prompt=loaded.agent.persona.mission,
        endpoint=provider.endpoint,
    )
    run_cli(loaded, llm, build_registry())


if __name__ == "__main__":
    main()
