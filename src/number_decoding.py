"""Constrained JSON number decoding."""

from __future__ import annotations

import re

from src.token_io import decode, encode, get_logits

MAX_VALUE_TOKENS = 64
_DIGIT_TOKEN = re.compile(r"^[0-9]+$")


def generate_number(
    model: object,
    ids: list[int],
    vocabulary: dict[int, str],
    separator: str,
    integer_only: bool = False,
) -> tuple[object, list[int]]:
    """Generate a JSON number token-by-token until the model stops it."""
    digit_tokens = _digit_tokens(vocabulary)
    minus_token = encode(model, "-")[0]
    dot_token = encode(model, ".")[0]
    stop_ids = encode(model, separator)
    stop_token = stop_ids[0]
    generated: list[int] = []
    seen_dot = False
    for step in range(MAX_VALUE_TOKENS):
        next_token = _best_number_token(
            model,
            ids,
            generated,
            digit_tokens,
            minus_token,
            dot_token,
            stop_token,
            step,
            seen_dot,
            integer_only,
        )
        if _has_digit(generated, digit_tokens) and next_token == stop_token:
            break
        if next_token == dot_token:
            seen_dot = True
        generated.append(next_token)
    number = _parse_number(decode(model, vocabulary, generated), seen_dot)
    return number, generated + stop_ids


def _best_number_token(
    model: object,
    ids: list[int],
    generated: list[int],
    digit_tokens: set[int],
    minus_token: int,
    dot_token: int,
    stop_token: int,
    step: int,
    seen_dot: bool,
    integer_only: bool,
) -> int:
    """Choose the highest-logit token that keeps a JSON number valid."""
    options = _number_options(
        generated,
        digit_tokens,
        minus_token,
        dot_token,
        stop_token,
        step,
        seen_dot,
        integer_only,
    )
    logits = get_logits(model, ids + generated)
    return max(options, key=lambda token_id: logits[token_id])


def _number_options(
    generated: list[int],
    digit_tokens: set[int],
    minus_token: int,
    dot_token: int,
    stop_token: int,
    step: int,
    seen_dot: bool,
    integer_only: bool,
) -> set[int]:
    """Return tokens allowed by a simple JSON-number grammar."""
    has_digit = _has_digit(generated, digit_tokens)
    options = set(digit_tokens)
    if step == 0:
        options.add(minus_token)
    if has_digit and not seen_dot and not integer_only:
        options.add(dot_token)
    if has_digit:
        options.add(stop_token)
    return options


def _digit_tokens(vocabulary: dict[int, str]) -> set[int]:
    """Return vocabulary tokens made only of digits."""
    return {
        token_id
        for token_id, text in vocabulary.items()
        if text and _DIGIT_TOKEN.match(text)
    }


def _has_digit(generated: list[int], digit_tokens: set[int]) -> bool:
    """Return whether generated tokens contain at least one digit token."""
    return any(token_id in digit_tokens for token_id in generated)


def _parse_number(text: str, seen_dot: bool) -> object:
    """Parse generated JSON number text, falling back to zero."""
    try:
        return float(text.strip()) if seen_dot else int(text.strip())
    except ValueError:
        return 0
