"""Thin CLI driver: read input -> run a turn -> print output -> prompt
inline for confirmations.

A client of ``AgentService`` — session lifecycle and the suspend/resume
cycle live there now, shared with the web UI, so the two clients cannot
drift. The CLI's own contribution is what it always was: a terminal
approver that asks the human directly (Rule 9).

Usage: python -m ui.cli <agent-name> [--env ENV]
"""

from __future__ import annotations

import argparse
from typing import Callable

from runtime import AgentService
from tools.defaults import build_registry

InputFn = Callable[[str], str]
PrintFn = Callable[[str], None]

__all__ = ["build_registry", "make_cli_approver", "run_cli", "main"]


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


def _print_activity(print_fn: PrintFn):
    def sink(event) -> None:
        if event.kind == "tool_start":
            print_fn(f"  … {event.tool}")

    return sink


def run_cli(
    service: AgentService,
    agent_name: str,
    input_fn: InputFn = input,
    print_fn: PrintFn = print,
) -> None:
    session_id = service.start_session(agent_name)
    session = service.session(session_id)
    approver = make_cli_approver(input_fn, print_fn)
    on_activity = _print_activity(print_fn)

    print_fn(f"agent '{service.agent_name(session_id)}' ready (autonomy: "
             f"{service.status(session_id)['autonomy']}). "
             f"'exit' or blank line to quit.")

    while True:
        try:
            line = input_fn("you> ")
        except EOFError:
            break
        if not line.strip() or line.strip().lower() == "exit":
            break

        update = service.send(session_id, line, on_activity=on_activity,
                              approver=approver)
        # With an inline approver, suspension is rare (e.g. piped stdin
        # closing); handle it anyway so the CLI can never dead-end.
        while update.status == "awaiting_approval":
            answer = input_fn(f"[pending] {update.pending.reason} — approve? [y/N] ")
            update = service.resolve_pending(
                session_id, answer.strip().lower() in ("y", "yes"),
                on_activity=on_activity, approver=approver,
            )

        if update.status == "completed":
            print_fn(update.text)
        elif update.status == "budget_exceeded":
            print_fn(f"[stopped: {update.detail}]")
            break
        elif update.status == "error":
            print_fn(f"[error] {update.detail}")
            break

    budget = service.status(session_id)["budget"]
    print_fn(
        f"[session {session.status}: tokens={budget['tokens']}, "
        f"tool_calls={budget['tool_calls']}, cost=${budget['cost_usd']:.4f}]"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run an agent at the terminal.")
    parser.add_argument("agent", help="agent profile name (e.g. scribe)")
    parser.add_argument("--env", default=None,
                        help="environment name (default: APP_ENV or 'dev')")
    args = parser.parse_args(argv)

    service = AgentService(env=args.env, registry=build_registry())
    run_cli(service, args.agent)


if __name__ == "__main__":
    main()
