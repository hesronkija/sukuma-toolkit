"""Corpus loaders for the verified Sukuma data sources.

Pillar A (text): AfriSpeech africa-corpus-builder output. The builder writes
one sentence per row; this loader accepts the CSV this project uses
(header ``suk``) as well as plain one-sentence-per-line text files.

Pillar B (speech): Sukuma Voices (sartifyllc/Sukuma-Voices-ACL on the
HuggingFace Hub). Loaded lazily through the ``datasets`` library so the
heavy dependency is only needed when audio is actually used.

License notes live in data/README.md: this package ships loaders and
download scripts, never redistributed corpus text.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from sukuma_toolkit.normalize import normalize_text


def load_afrispeech_text(path: str | Path) -> list[str]:
    """Load Sukuma sentences from an AfriSpeech corpus-builder export.

    Accepts a CSV with a ``suk`` header column (the format produced for
    this project) or a plain text file with one sentence per line.
    Returns normalized, non-empty sentences in file order.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"corpus file not found: {path}")

    sentences: list[str] = []
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "suk" in reader.fieldnames:
                rows: Iterator[dict] = reader
                get = lambda r: r.get("suk", "")  # noqa: E731
            else:
                f.seek(0)
                rows = ({0: line} for line in f)
                get = lambda r: r[0]  # noqa: E731
            for row in rows:
                s = normalize_text(get(row))
                if s:
                    sentences.append(s)
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            s = normalize_text(line)
            if s:
                sentences.append(s)
    return sentences


def load_sukuma_voices(split: str = "train", streaming: bool = False):
    """Load the Sukuma Voices speech corpus from the HuggingFace Hub.

    Requires the ``datasets`` package (and ``librosa``/``soundfile`` for
    audio decoding, via the ``audio`` extra). Splits: "train", "test",
    "test_indistribution_synthesis", "test_outdistribution_synthesis".

    Each example has ``audio`` (dict with array/sampling_rate/path) and
    ``text`` (the Sukuma transcription).
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "load_sukuma_voices needs the 'datasets' package: "
            "pip install 'sukuma-toolkit[audio]'"
        ) from exc

    valid = {
        "train",
        "test",
        "test_indistribution_synthesis",
        "test_outdistribution_synthesis",
    }
    if split not in valid:
        raise ValueError(f"split must be one of {sorted(valid)}, got {split!r}")
    return load_dataset(
        "sartifyllc/Sukuma-Voices-ACL", split=split, streaming=streaming
    )


def iter_transcripts(dataset) -> Iterator[str]:
    """Yield normalized transcripts from a Sukuma Voices dataset split."""
    for ex in dataset:
        text = normalize_text(ex["text"])
        if text:
            yield text
