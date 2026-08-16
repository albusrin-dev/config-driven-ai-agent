"""Window adapter: head + compact marker + recent tail, pair-integrity safe.

When the conversation fits the budget it is returned unchanged. Otherwise:
- Head (always preserved): leading system message(s) and the initial task
  (first user message) — the model must never lose who it is or what it
  was asked to do.
- Tail: the most recent turns that fit the remaining budget, verbatim.
- Middle: replaced by one compact marker message.

Integrity: cuts happen only at block boundaries, where a block is an
assistant message with tool_calls together with its following tool_result
messages — a tool_use is never split from its result.

Deterministic, no LLM call, no budget cost, and the input list is never
mutated (Rule 11: stored history is the session's, untouched).

KNOWN LIMITATION: dropping the middle can lose intermediate steps on very
long tasks, so the model may repeat work or lose thread. Acceptable first
pass — strictly better than hitting the token wall. The flagged upgrade is
an LLM-summarizing adapter behind this same port, built when drop-middle
demonstrably loses needed information.
"""

from __future__ import annotations

from core.memory import ContextBudgetError, MemoryPort, TokenCounter


def _marker(omitted_count: int) -> dict:
    return {
        "role": "user",
        "content": f"[{omitted_count} earlier messages omitted to fit the context window]",
    }


def _split_head(conversation: list[dict]) -> tuple[list[dict], list[dict]]:
    """Head = leading system message(s) + the first user message."""
    i = 0
    while i < len(conversation) and conversation[i]["role"] == "system":
        i += 1
    if i < len(conversation) and conversation[i]["role"] == "user":
        i += 1
    return conversation[:i], conversation[i:]


def _group_blocks(messages: list[dict]) -> list[list[dict]]:
    """Atomic blocks: an assistant-with-tool_calls plus its tool_results
    stay together; everything else is a singleton block."""
    blocks: list[list[dict]] = []
    open_block = False
    for message in messages:
        if message["role"] == "tool_result" and open_block:
            blocks[-1].append(message)
        elif message["role"] == "assistant" and message.get("tool_calls"):
            blocks.append([message])
            open_block = True
        else:
            blocks.append([message])
            open_block = False
    return blocks


class WindowMemory(MemoryPort):
    def assemble_context(
        self,
        conversation: list[dict],
        budget_tokens: int,
        count_tokens: TokenCounter,
    ) -> list[dict]:
        convo = [dict(message) for message in conversation]  # never mutate input
        if count_tokens(convo) <= budget_tokens:
            return convo

        head, rest = _split_head(convo)
        blocks = _group_blocks(rest)

        # Conservative marker estimate: sized with the largest possible k,
        # so the final (smaller-k) marker can only shrink the total.
        marker_estimate = _marker(len(rest))

        tail: list[dict] = []
        for block in reversed(blocks):
            candidate = block + tail
            if count_tokens(head + [marker_estimate] + candidate) <= budget_tokens:
                tail = candidate
            else:
                break

        if not tail:
            raise ContextBudgetError(
                f"cannot fit context: head ({len(head)} message(s)) plus the "
                f"most recent turn exceeds budget_tokens={budget_tokens}; "
                f"raise the budget or shorten the messages"
            )

        omitted = len(rest) - len(tail)
        return head + [_marker(omitted)] + tail
