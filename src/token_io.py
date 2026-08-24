"""Low-level helpers for talking to the Small_LLM_Model SDK."""

from __future__ import annotations

from collections.abc import Callable
from typing import SupportsInt, cast


def encode(model: object, text: str) -> list[int]:
    """converts text into token IDs."""
    encoder = getattr(model, "encode", None)
    if not callable(encoder):
        raise TypeError("model does not provide encode")
    typed_encoder = cast(Callable[[str], object], encoder)
    return _ids_from_encoded(typed_encoder(text))


def get_logits(model: object, input_ids: list[int]) -> list[float]:
    """gets the model scores for possible next tokens."""
    logits_getter = getattr(model, "get_logits_from_input_ids", None)
    if not callable(logits_getter):
        raise TypeError("model does not provide get_logits_from_input_ids")
    typed_getter = cast(Callable[[list[int]], list[float]], logits_getter)
    return typed_getter(input_ids)


def decode(
    model: object, vocabulary: dict[int, str], token_ids: list[int]
) -> str:
    """converts token IDs back into text."""
    decoder = getattr(model, "decode", None)
    if callable(decoder):
        return str(decoder(token_ids))
    text = "".join(vocabulary.get(token_id, "") for token_id in token_ids)
    return text.replace("Ġ", " ").replace("▁", " ")


def _ids_from_encoded(encoded: object) -> list[int]:
    """the encoder output may have different formats depending on the SDK.
    this function normalizes all supported
    formats into a plain list of integer token IDs"""
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
