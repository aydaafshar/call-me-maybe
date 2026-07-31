"""This file lets the model choose only from predefined options by generating
 one token at a time and.
 eliminating impossible choices after each generated token."""

from __future__ import annotations

from src.token_io import encode, get_logits


def choose_literal(
    model: object,
    prefix_ids: list[int],
    options: list[str],
) -> tuple[str, list[int]]:
    """Pick the option whose tokenization the model scores highest."""
    tokenized = [encode(model, option) for option in options]
    generated: list[int] = []
    active = list(range(len(options)))
    while True:
        if len(active) == 1 or not active:
            chosen = active[0] if active else 0
            return options[chosen], tokenized[chosen]
        finished = [i for i in active if len(generated) == len(tokenized[i])]
        if finished:
            chosen = finished[0]
            return options[chosen], tokenized[chosen]
        next_token = _best_next_literal_token(
            model, prefix_ids, generated, tokenized, active
        )
        generated.append(next_token)
        active = _matching_options(tokenized, active, generated)


def _best_next_literal_token(
    model: object,
    prefix_ids: list[int],
    generated: list[int],
    tokenized: list[list[int]],
    active: list[int],
) -> int:
    """Choose the highest-logit token and to do this one we use set."""
    valid_next = {
        tokenized[i][len(generated)]
        for i in active
        if len(generated) < len(tokenized[i])
    }
    logits = get_logits(model, prefix_ids + generated)
    return max(valid_next, key=lambda token_id: logits[token_id])


def _matching_options(
    tokenized: list[list[int]],
    active: list[int],
    generated: list[int],
) -> list[int]:
    """Keep options that still match. with de generated ones."""
    return [
        i
        for i in active
        if len(generated) <= len(tokenized[i])
        and tokenized[i][: len(generated)] == generated
    ]
