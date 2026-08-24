"""Vocabulary loading for constrained decoding.
The vocabulary connects token IDs to text.
It supports different vocabulary file formats.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_vocabulary(path: Path) -> dict[int, str]:

    with path.open("r", encoding="utf-8") as vocab_file:
        raw = json.load(vocab_file)
    if isinstance(raw, dict) and isinstance(raw.get("model"), dict):
        model_vocab = raw["model"].get("vocab")
        if isinstance(model_vocab, dict):
            raw = model_vocab
    if not isinstance(raw, dict) or not raw:
        raise TypeError("unsupported vocabulary file format")
    sample_value = next(iter(raw.values()))
    if isinstance(sample_value, int):
        return {
            token_id: str(token_text) for token_text, token_id in raw.items()
        }
    return {
        int(token_id): str(token_text) for token_id, token_text in raw.items()
    }
