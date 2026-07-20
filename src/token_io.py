"""Low-level helpers for talking to the Small_LLM_Model SDK."""

from __future__ import annotations

from collections.abc import Callable
from typing import SupportsInt, cast


def encode(model: object, text: str) -> list[int]:
    """Encode text using the public SDK method."""
    encoder = getattr(model, "encode", None)
    if not callable(encoder):
        raise TypeError("model does not provide encode")
    typed_encoder = cast(Callable[[str], object], encoder)
    return _ids_from_encoded(typed_encoder(text))


def get_logits(model: object, input_ids: list[int]) -> list[float]:
    """Get next-token logits using the public SDK method."""
    logits_getter = getattr(model, "get_logits_from_input_ids", None)
    if not callable(logits_getter):
        raise TypeError("model does not provide get_logits_from_input_ids")
    typed_getter = cast(Callable[[list[int]], list[float]], logits_getter)
    return typed_getter(input_ids)


def decode(model: object, vocabulary: dict[int, str], token_ids: list[int]) -> str:
    """Decode token ids to text, using the SDK's optional decode method."""
    decoder = getattr(model, "decode", None)
    if callable(decoder):
        return str(decoder(token_ids))
    text = "".join(vocabulary.get(token_id, "") for token_id in token_ids)
    return text.replace("Ġ", " ").replace("▁", " ")


def _ids_from_encoded(encoded: object) -> list[int]:
    """Normalize SDK encodings to a plain list of token IDs."""
    if isinstance(encoded, list):
        if encoded and isinstance(encoded[0], list):
            return [int(item) for item in cast(list[SupportsInt], encoded[0])]
        return [int(cast(SupportsInt, item)) for item in encoded]
    if hasattr(encoded, "tolist"):
        raw = encoded.tolist()
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            return [int(cast(SupportsInt, item)) for item in raw[0]]
        if isinstance(raw, list):
            return [int(cast(SupportsInt, item)) for item in raw]
    raise TypeError("unsupported token encoding returned by llm_sdk")
