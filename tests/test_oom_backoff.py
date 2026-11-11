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


def _clips(n: int, seconds: float = 0.1) -> list[dict]:
    return [{"id": str(i), "waveform": np.zeros(int(16000 * seconds), dtype="float32"), "sample_rate": 16000,
             "text": "a"} for i in range(n)]


class FakeWhisper:
    """Echoes each clip's length; fails like the real pipeline on OOM or on mixed short/long batches."""

    def __init__(self, fits: int = 99):
        self.fits, self.calls = fits, []

    def __call__(self, inputs, batch_size, generate_kwargs):
        self.calls.append(batch_size)
        if batch_size > self.fits:
            raise RuntimeError("CUDA out of memory. Tried to allocate 294.00 MiB")
        lengths = [len(x["raw"]) / x["sampling_rate"] for x in inputs]
        if batch_size > 1 and any(n > 30 for n in lengths):
            raise ValueError("The elements of the batch contain different keys. Cannot batch them")
        return [{"text": f" {n:.0f}s "} for n in lengths]


def test_whisper_halves_batch_on_oom(capsys):
    mod = _load("whisper_baseline")
    pipe = FakeWhisper(fits=2)
    hyps = mod.transcribe(pipe, _clips(40), batch_size=8)
    assert len(hyps) == 40 and hyps[0] == "0s"
    assert pipe.calls[:3] == [8, 4, 2] and set(pipe.calls[3:]) == {2}  # later chunks keep the smaller size
    assert "retrying with batch size 2" in capsys.readouterr().out


def test_whisper_runs_long_clips_alone_and_keeps_order():
    mod = _load("whisper_baseline")
    clips = _clips(3, 10) + _clips(1, 35) + _clips(2, 12) + _clips(1, 38)
    pipe = FakeWhisper()
    hyps = mod.transcribe(pipe, clips, batch_size=4)
    assert hyps == ["10s", "10s", "10s", "35s", "12s", "12s", "38s"]
    assert pipe.calls == [4, 1]  # short clips batched by 4, long clips in a batch of 1


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
