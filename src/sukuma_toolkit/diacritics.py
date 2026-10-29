"""Diacritic restoration for Sukuma: turn plain text (wandijo) into diacritic form (wandījo).

Two layers, both learned from text that already has diacritics:

1. **Word table.** For a word seen in training, use its most frequent written
   form, or the most frequent form after the same previous word when the plain
   spelling is ambiguous.
2. **Character model.** For an unseen word, decide each vowel separately from
   its surrounding letters, backing off from 3 letters of context on each side
   to none.

Capital letters and punctuation are kept as they were. The model is plain JSON,
so it can be inspected, versioned and shipped without extra dependencies.

    >>> r = DiacriticRestorer().fit(["Ū Yesu ūbawīla", "Ū Yesu alīyo"])
    >>> r.restore("U Yesu ubawila")
    'Ū Yesu ūbawīla'
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from sukuma_toolkit.normalize import detect_diacritics, normalize_text, strip_diacritics

_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
VOWELS = set("aeiou")
MAX_CONTEXT = 3
FORMAT_VERSION = 1


def _words(text: str) -> list[str]:
    return _WORD.findall(normalize_text(text).lower())


def _match_case(source: str, restored: str) -> str:
    """Copy the capitalization of ``source`` onto ``restored`` (same length)."""
    if source.islower():
        return restored
    if source.isupper() and len(source) > 1:
        return restored.upper()
    return "".join(r.upper() if s.isupper() else r for s, r in zip(source, restored))


class DiacriticRestorer:
    """Learns how plain Sukuma words are written with diacritics."""

    def __init__(self) -> None:
        self.words: dict[str, Counter] = defaultdict(Counter)
        self.bigrams: dict[str, Counter] = defaultdict(Counter)
        self.chars: dict[str, Counter] = defaultdict(Counter)

    # ------------------------------------------------------------------ training
    def fit(self, sentences: Iterable[str], require_diacritics: bool = True) -> "DiacriticRestorer":
        """Learn from sentences written with diacritics.

        Sentences without any diacritic are skipped by default: they come from
        the plain orthography and would teach the model to drop marks.
        """
        for sentence in sentences:
            if require_diacritics and not detect_diacritics(sentence):
                continue
            prev = "<s>"
            for word in _words(sentence):
                plain = strip_diacritics(word)
                if len(plain) != len(word):  # decomposed or unusual marks: skip, keep alignment exact
                    prev = plain
                    continue
                self.words[plain][word] += 1
                self.bigrams[f"{prev} {plain}"][word] += 1
                padded = f"^{plain}$"
                for i, ch in enumerate(plain):
                    if ch in VOWELS:
                        target = word[i]
                        for k in range(MAX_CONTEXT + 1):
                            self.chars[self._ctx(padded, i + 1, k)][target] += 1
                prev = plain
        return self

    @staticmethod
    def _ctx(padded: str, pos: int, k: int) -> str:
        return f"{k}|{padded[max(0, pos - k):pos]}_{padded[pos + 1:pos + 1 + k]}|{padded[pos]}"

    # --------------------------------------------------------------- prediction
    def restore_word(self, word: str, prev: str = "<s>") -> str:
        """Restore one word (any case); ``prev`` is the previous plain lowercase word."""
        lower = strip_diacritics(word.lower())
        if lower in self.words:
            forms = self.words[lower]
            if len(forms) > 1 and f"{prev} {lower}" in self.bigrams:
                forms = self.bigrams[f"{prev} {lower}"]
            return _match_case(word, forms.most_common(1)[0][0])
        return _match_case(word, self._restore_unseen(lower))

    def _restore_unseen(self, plain: str) -> str:
        padded = f"^{plain}$"
        out = list(plain)
        for i, ch in enumerate(plain):
            if ch not in VOWELS:
                continue
            for k in range(MAX_CONTEXT, -1, -1):
                counts = self.chars.get(self._ctx(padded, i + 1, k))
                if counts:
                    out[i] = counts.most_common(1)[0][0]
                    break
        return "".join(out)

    def restore(self, text: str) -> str:
        """Restore diacritics in running text, keeping punctuation, spacing and case."""
        text = unicodedata.normalize("NFC", text)
        prev = ["<s>"]

        def repl(m: re.Match) -> str:
            restored = self.restore_word(m.group(0), prev[0])
            prev[0] = strip_diacritics(m.group(0).lower())
            return restored

        return _WORD.sub(repl, text)

    def is_known(self, word: str) -> bool:
        return strip_diacritics(word.lower()) in self.words

    # -------------------------------------------------------------- persistence
    def to_dict(self, min_count: int = 1) -> dict:
        def dump(table: dict[str, Counter]) -> dict:
            return {k: dict(v) for k, v in table.items() if sum(v.values()) >= min_count}

        bigrams = {k: dict(v) for k, v in self.bigrams.items() if len(self.words.get(k.split(" ", 1)[1], ())) > 1}
        return {"format": FORMAT_VERSION, "words": dump(self.words), "bigrams": bigrams, "chars": dump(self.chars)}

    @classmethod
    def from_dict(cls, data: dict) -> "DiacriticRestorer":
        if data.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported restorer format: {data.get('format')!r}")
        r = cls()
        for name in ("words", "bigrams", "chars"):
            table = getattr(r, name)
            for k, v in data[name].items():
                table[k] = Counter(v)
        return r

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DiacriticRestorer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def evaluate_restorer(restorer: DiacriticRestorer, sentences: Iterable[str]) -> dict:
    """Strip diacritics from reference sentences, restore them, and score the result.

    Word accuracy counts a word right only if every mark matches. ``baseline``
    is the score of leaving the plain text as is.
    """
    total = correct = baseline = marked = marked_correct = unseen = unseen_correct = 0
    for sentence in sentences:
        ref = _words(sentence)
        hyp = _words(restorer.restore(strip_diacritics(normalize_text(sentence))))
        if len(ref) != len(hyp):
            continue
        for r, h in zip(ref, hyp):
            total += 1
            correct += r == h
            baseline += r == strip_diacritics(r)
            if r != strip_diacritics(r):
                marked += 1
                marked_correct += r == h
            if not restorer.is_known(r):
                unseen += 1
                unseen_correct += r == h
    return {
        "words": total,
        "word_accuracy": correct / total if total else 0.0,
        "baseline_accuracy": baseline / total if total else 0.0,
        "marked_words": marked,
        "marked_word_accuracy": marked_correct / marked if marked else 0.0,
        "unseen_words": unseen,
        "unseen_word_accuracy": unseen_correct / unseen if unseen else 0.0,
    }


DEFAULT_MODEL = "restorer_speech_corpus.json"


def load_default() -> DiacriticRestorer:
    """The restorer shipped with the package.

    Trained on the Sukuma Speech Corpus prompt sentences (joseph-telemala,
    CC BY 4.0), which follow that corpus's spelling convention. For Bible-style
    text, train your own with ``scripts/train_restorer.py`` and pass it to the CLI
    with ``--model``.
    """
    from importlib.resources import files

    return DiacriticRestorer.from_dict(json.loads((files("sukuma_toolkit") / "data" / DEFAULT_MODEL).read_text("utf-8")))
