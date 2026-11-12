#!/usr/bin/env python3
"""Fine-tune NLLB-200 to translate Sukuma -> Swahili.

    python scripts/build_translation_data.py        # data/raw/mt/*.tsv (local only)
    python scripts/train_translator.py              # GPU; see notebooks/colab_translation.ipynb

NLLB-200 already translates Swahili and many Bantu languages but has no
Sukuma. This adds a new language token, suk_Latn, starts its embedding from
Swahili's, and fine-tunes facebook/nllb-200-distilled-600M on the Bible verse
pairs. Half of the Sukuma sources are shown with their diacritics stripped, so
the model handles both written forms (the UDHR test set is plain).

About 25 minutes per epoch on a Colab L4. The trained model is derived from
copyrighted Bible text: keep it private (Drive, or a private Hub repo).

Writes the model and tokenizer to --out (default models/nllb-suk-swh).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sukuma_toolkit.normalize import strip_diacritics  # noqa: E402
from sukuma_toolkit.translate import SUKUMA_CODE, SWAHILI_CODE  # noqa: E402

BASE_MODEL = "facebook/nllb-200-distilled-600M"


def read_tsv(path: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def add_sukuma_token(tokenizer, model) -> int:
    """Add suk_Latn as a language token, initialised from swh_Latn's embedding."""
    import torch

    if SUKUMA_CODE not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": tokenizer.additional_special_tokens + [SUKUMA_CODE]},
                                     replace_additional_special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        suk, swh = tokenizer.convert_tokens_to_ids([SUKUMA_CODE, SWAHILI_CODE])
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            emb[suk] = emb[swh]
    tokenizer.src_lang, tokenizer.tgt_lang = SUKUMA_CODE, SWAHILI_CODE
    return tokenizer.convert_tokens_to_ids(SUKUMA_CODE)


class PairDataset:
    """Tokenizes on the fly; strips the source's diacritics with probability ``plain_share``."""

    def __init__(self, rows: list[dict], tokenizer, max_len: int = 128, plain_share: float = 0.5, seed: int = 0):
        self.rows, self.tok, self.max_len, self.plain_share = rows, tokenizer, max_len, plain_share
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        src = strip_diacritics(r["suk"]) if self.rng.random() < self.plain_share else r["suk"]
        return self.tok(src, text_target=r["swh"], truncation=True, max_length=self.max_len)


def train(data_dir: Path, out_dir: Path, base_model: str = BASE_MODEL, epochs: float = 1.0, batch_size: int = 16,
          lr: float = 1e-4, max_steps: int = -1, max_len: int = 128, limit: int = 0, seed: int = 42,
          checkpoints: Path | None = None) -> Path:
    import torch
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    add_sukuma_token(tokenizer, model)
    train_rows = read_tsv(data_dir / "train.tsv")
    dev_rows = read_tsv(data_dir / "dev.tsv")
    if limit:
        train_rows, dev_rows = train_rows[:limit], dev_rows[:max(2, limit // 10)]
    print(f"{len(train_rows):,} training pairs, {len(dev_rows):,} dev pairs")
    gpu = torch.cuda.is_available()
    args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoints or out_dir / "checkpoints"),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        bf16=gpu and torch.cuda.is_bf16_supported(),
        eval_strategy="steps" if max_steps < 0 or max_steps >= 1000 else "no",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=1,
        logging_steps=100,
        report_to=[],
        seed=seed,
        dataloader_num_workers=0,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=PairDataset(train_rows, tokenizer, max_len, seed=seed),
        eval_dataset=PairDataset(dev_rows, tokenizer, max_len, plain_share=0.0),
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
    )
    trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Model saved to {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "mt")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "nllb-suk-swh")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop early (smoke tests)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training pairs (smoke tests)")
    ap.add_argument("--checkpoints", type=Path, help="where to keep mid-training checkpoints (default: <out>/checkpoints)")
    args = ap.parse_args(argv)
    train(args.data, args.out, args.base_model, args.epochs, args.batch_size, args.lr, args.max_steps,
          limit=args.limit, checkpoints=args.checkpoints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
