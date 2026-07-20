# ABOUTME: LLM SDK for local model inference using Hugging Face transformers.
# ABOUTME: Provides Small_LLM_Model wrapper class.

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import cast

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import logging
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer


logging.set_verbosity_error()  # keep the console clean


class Small_LLM_Model:
    """Utility class wrapping a lightweight Hugging Face causal-LM.

    Parameters
    ----------
    model_name: str, default="Qwen/Qwen3-0.6B"
        Identifier of the model on the HF Hub.
    device: str | None, default=None
        Computation device. If *None* selects ``mps``, ``cuda``, then ``cpu``.
    dtype: torch.dtype | None, default=None
        Numerical precision. GPU and MPS default to ``float16``.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_name = model_name

        # Auto-select device with priority: mps > cuda > cpu
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        if dtype is None:
            dtype = torch.float16 if self._device in ["cuda", "mps"] else torch.float32
        self._dtype = dtype

        # Load tokenizer and model.
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            # ensure we have a pad token to keep batch helpers happy
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
        self._model.to(self._device)
        self._model.eval()

        # switch to inference-only mode
        for p in self._model.parameters():
            p.requires_grad = False

    def encode(self, text: str) -> torch.Tensor:
        """Tokenise text and return a 2-D tensor on the target device."""
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self._device, dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Inverse of :py:meth:`encode`. Removes special tokens."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return cast(str, self._tokenizer.decode(ids, skip_special_tokens=True))

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return raw next-token logits for a list of token IDs."""
        input_tensor = torch.tensor([input_ids], device=self._device, dtype=torch.long)
        with torch.no_grad():
            out = self._model(input_ids=input_tensor)
        # Get logits for the last token in the sequence.
        logits = out.logits[0, -1].tolist()
        return [float(x) for x in logits]

    def get_path_to_vocabulary_json(self) -> str:
        """Return the local path to a token_id -> token_text vocabulary file."""
        cache_name = self._model_name.replace("/", "_") + "_vocabulary.json"
        cache_path = Path(tempfile.gettempdir()) / cache_name
        if not cache_path.exists():
            vocab = self._tokenizer.get_vocab()
            vocabulary = {
                token_id: token_text for token_text, token_id in vocab.items()
            }
            cache_path.write_text(json.dumps(vocabulary), encoding="utf-8")
        return str(cache_path)

    def get_path_to_vocab_file(self) -> str:
        """Return the local path to the tokenizer vocabulary file."""
        vocab_file_name = self._tokenizer.vocab_files_names.get(
            "vocab_file", "vocab.json"
        )
        vocab_path = hf_hub_download(
            repo_id=self._model_name,
            filename=vocab_file_name
        )
        return cast(str, vocab_path)

    def get_path_to_merges_file(self) -> str:
        """Return the local path to the tokenizer merges file."""
        merges_file_name = self._tokenizer.vocab_files_names.get(
            "merges_file", "merges.txt"
        )
        merges_path = hf_hub_download(
            repo_id=self._model_name,
            filename=merges_file_name
        )
        return cast(str, merges_path)

    def get_path_to_tokenizer_file(self) -> str:
        """Return the local path to the tokenizer JSON file."""
        tokenizer_file_name = self._tokenizer.vocab_files_names.get(
            "tokenizer_file", "tokenizer.json"
        )
        tokenizer_path = hf_hub_download(
            repo_id=self._model_name,
            filename=tokenizer_file_name
        )
        return cast(str, tokenizer_path)
