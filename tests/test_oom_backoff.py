"""The GPU benchmark scripts halve the batch size and retry when the GPU runs out of memory."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
np = pytest.importorskip("numpy")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_oom_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _clips(n: int) -> list[dict]:
    return [{"id": str(i), "waveform": np.zeros(1600, dtype="float32"), "sample_rate": 16000, "text": "a"}
            for i in range(n)]


class FakeWhisper:
    def __init__(self, fits: int):
        self.fits, self.calls = fits, []

    def __call__(self, inputs, batch_size, generate_kwargs):
        self.calls.append(batch_size)
        if batch_size > self.fits:
            raise RuntimeError("CUDA out of memory. Tried to allocate 294.00 MiB")
        return [{"text": f" hyp{i} "} for i in range(len(inputs))]


def test_whisper_halves_batch_on_oom(capsys):
    mod = _load("whisper_baseline")
    pipe = FakeWhisper(fits=2)
    hyps = mod.transcribe(pipe, _clips(40), batch_size=8)
    assert len(hyps) == 40 and hyps[0] == "hyp0"
    assert pipe.calls[:3] == [8, 4, 2] and set(pipe.calls[3:]) == {2}  # later chunks keep the smaller size
    assert "retrying with batch size 2" in capsys.readouterr().out


def test_whisper_reraises_other_errors_and_oom_at_batch_one():
    mod = _load("whisper_baseline")

    def broken(inputs, batch_size, generate_kwargs):
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        mod.transcribe(broken, _clips(2), batch_size=4)
    with pytest.raises(RuntimeError):
        mod.transcribe(FakeWhisper(fits=0), _clips(2), batch_size=4)


def test_omnilingual_halves_batch_on_oom(tmp_path):
    pytest.importorskip("torch")
    mod = _load("omnilingual_smoke_test")

    class FakeOmni:
        calls: list[int] = []

        def transcribe(self, inputs, lang, batch_size):
            self.calls.append(batch_size)
            if batch_size > 1:
                raise RuntimeError("CUDA out of memory")
            return ["a"] * len(inputs)

    fake = FakeOmni()
    r = mod.run(_clips(3), fake, "omniASR_CTC_300M_v2", tmp_path, batch_size=4)
    assert r["clips"] == 3 and fake.calls == [4, 2, 1]
