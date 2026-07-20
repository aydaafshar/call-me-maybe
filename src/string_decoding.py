"""Constrained JSON string decoding."""

from __future__ import annotations

from src.token_io import decode, encode, get_logits

MAX_VALUE_TOKENS = 64


def generate_string(
    model: object,
    ids: list[int],
    vocabulary: dict[int, str],
    separator: str,
) -> tuple[str, list[int]]:
    """Generate free-form string content until the model closes the quote."""
    open_ids = encode(model, '"')
    close_ids = encode(model, '"' + separator)
    close_token = close_ids[0]
    context = ids + open_ids
    generated: list[int] = []
    safe_tokens = _safe_string_tokens(vocabulary)
    for _ in range(MAX_VALUE_TOKENS):
        next_token = _best_string_token(
            model, context, generated, safe_tokens, close_token
        )
        if next_token == close_token:
            break
        generated.append(next_token)
        if is_repeating(generated):
            break
    content = decode(model, vocabulary, generated)
    return content, open_ids + generated + close_ids


def is_repeating(
    generated: list[int],
    max_period: int = 12,
    min_cycles: int = 3,
) -> bool:
    """Detect a small-model generation loop repeating the same token cycle."""
    for period in range(1, max_period + 1):
        window = period * min_cycles
        if len(generated) < window:
            continue
        block = generated[-period:]
        cycles = generated[-window:]
        chunks = (cycles[i * period: (i + 1) * period] for i in range(min_cycles))
        if all(chunk == block for chunk in chunks):
            return True
    return False


def _safe_string_tokens(vocabulary: dict[int, str]) -> set[int]:
    """Return tokens that cannot directly break a JSON string."""
    return {
        token_id
        for token_id, text in vocabulary.items()
        if text and '"' not in text and "\\" not in text and "\n" not in text
    }


def _best_string_token(
    model: object,
    context: list[int],
    generated: list[int],
    safe_tokens: set[int],
    close_token: int,
) -> int:
    """Choose the highest-logit valid string token."""
    logits = get_logits(model, context + generated)
    valid = set(safe_tokens)
    valid.add(close_token)
    return max(valid, key=lambda token_id: logits[token_id])
