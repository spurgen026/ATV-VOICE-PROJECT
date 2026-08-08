"""stt.py resilience tests. load_model()'s CPU fallback is the single
highest-value resilience fix from this round (see stt.py's docstring) -
this app's target hardware (Pi 5) has no GPU at all, so a GPU-load
failure must degrade gracefully, not crash the whole app at startup.
Tested by making a fake WhisperModel fail on the configured device and
succeed on CPU, rather than needing to actually break CUDA to prove it.
"""

import numpy as np
import pytest

from resort_atv_voice import stt


class _FakeModel:
    def __init__(self, model_size, device, compute_type):
        self.device = device
        self.compute_type = compute_type


def test_load_model_falls_back_to_cpu_when_configured_device_fails(monkeypatch):
    def fake_whisper_model(model_size, device, compute_type):
        if device == stt.WHISPER_DEVICE:
            raise RuntimeError("simulated GPU load failure")
        return _FakeModel(model_size, device, compute_type)

    monkeypatch.setattr(stt, "WhisperModel", fake_whisper_model)
    monkeypatch.setattr(stt, "WHISPER_DEVICE", "cuda")

    model = stt.load_model()
    assert model.device == "cpu"
    assert model.compute_type == stt.CPU_FALLBACK_COMPUTE_TYPE


def test_load_model_does_not_swallow_a_cpu_failure(monkeypatch):
    # If CPU itself is the configured device and loading fails, there's
    # nowhere left to fall back to - the failure must propagate, not be
    # silently swallowed into some other broken state.
    def always_fails(model_size, device, compute_type):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(stt, "WhisperModel", always_fails)
    monkeypatch.setattr(stt, "WHISPER_DEVICE", "cpu")

    with pytest.raises(RuntimeError):
        stt.load_model()


def test_load_model_succeeds_without_fallback_when_configured_device_works(monkeypatch):
    monkeypatch.setattr(stt, "WhisperModel", _FakeModel)
    monkeypatch.setattr(stt, "WHISPER_DEVICE", "cuda")

    model = stt.load_model()
    assert model.device == "cuda"


class _FakeInputStream:
    """Immediately reports silence, so record_query() hits its
    speech_timeout_ms path and returns fast without a real audio device."""

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, num_samples):
        return np.zeros((num_samples, 1), dtype="int16"), False


def test_record_query_settles_before_opening_the_input_stream(monkeypatch):
    # Echo/feedback mitigation (see MIC_SETTLE_MS in config.py) only works
    # if the settle pause happens BEFORE the mic starts listening, not
    # after - this is exactly the kind of ordering bug that's easy to
    # introduce silently in a later refactor.
    call_order = []

    def fake_sleep(seconds):
        call_order.append(("sleep", seconds))

    def fake_input_stream(**kwargs):
        call_order.append(("stream_open",))
        return _FakeInputStream(**kwargs)

    monkeypatch.setattr(stt.time, "sleep", fake_sleep)
    monkeypatch.setattr(stt.sd, "InputStream", fake_input_stream)

    stt.record_query(speech_timeout_ms=stt.QUERY_CHUNK_MS)

    assert call_order[0] == ("sleep", stt.MIC_SETTLE_MS / 1000)
    assert call_order[1] == ("stream_open",)


def test_record_query_returns_empty_array_when_nothing_is_heard(monkeypatch):
    monkeypatch.setattr(stt.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(stt.sd, "InputStream", _FakeInputStream)

    audio = stt.record_query(speech_timeout_ms=stt.QUERY_CHUNK_MS)
    assert audio.size == 0
