"""Sukuma -> Swahili translation with a fine-tuned NLLB model.

NLLB-200 (Meta) translates Swahili but has no Sukuma. ``scripts/train_translator.py``
teaches it Sukuma by adding a ``suk_Latn`` language token and fine-tuning on
Sukuma-Swahili sentence pairs. This module loads such a model and translates.

    >>> t = Translator("models/nllb-suk-swh")              # doctest: +SKIP
    >>> t.translate(["Banhu bose bakabyalagwa na wiyabi."])  # doctest: +SKIP

Needs ``pip install sukuma-toolkit[translate]`` (torch, transformers, sentencepiece).
The toolkit ships no trained model: the only large Sukuma-Swahili parallel text
is copyrighted Bible translations, so a model trained on it stays private.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from sukuma_toolkit.normalize import normalize_text

SUKUMA_CODE = "suk_Latn"
SWAHILI_CODE = "swh_Latn"
MODEL_ENV = "SUKUMA_MT_MODEL"


class Translator:
    """Wraps a fine-tuned NLLB checkpoint for batch Sukuma -> Swahili translation."""

    def __init__(self, model_dir: str | Path | None = None, device: str | None = None, num_beams: int = 4,
                 src_lang: str = SUKUMA_CODE, tgt_lang: str = SWAHILI_CODE) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_dir = model_dir or os.environ.get(MODEL_ENV)
        if not model_dir:
            raise ValueError(f"no model: pass a model directory or set ${MODEL_ENV}")
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.tokenizer.src_lang = src_lang
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), torch_dtype=dtype).to(self.device).eval()
        self.tgt_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        self.num_beams = num_beams

    def translate(self, texts: Sequence[str], batch_size: int = 16, max_new_tokens: int = 256) -> list[str]:
        import torch

        out: list[str] = []
        for start in range(0, len(texts), batch_size):
            batch = [normalize_text(t) for t in texts[start:start + batch_size]]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                 max_length=256).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, forced_bos_token_id=self.tgt_id, num_beams=self.num_beams,
                                          max_new_tokens=max_new_tokens)
            out += [s.strip() for s in self.tokenizer.batch_decode(gen, skip_special_tokens=True)]
        return out
