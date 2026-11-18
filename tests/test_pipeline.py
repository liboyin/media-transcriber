"""Tests for the pipeline that the worker and the benchmark share, with a fake model."""

import wave
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from fake_engine import FakeModel

import media_transcriber.pipeline as testee
from media_transcriber.audio import EmptyAudioError
from media_transcriber.engine import Segment
from media_transcriber.rejection import REJECTED_SUBSTRINGS, RejectedTranscriptionError
from media_transcriber.srt import segments_to_srt

SEGMENTS = [Segment(0.0, 1.0, 'hello'), Segment(1.0, 2.0, 'hello'), Segment(2.0, 3.5, 'world')]


def stepping_clock(*readings: float) -> Callable[[], float]:
    """Return a clock that yields the given readings in order."""
    values: Iterator[float] = iter(readings)
    return lambda: next(values)


@dataclass
class FakeDecoding:
    """The fixed waveform that stands in for decoding, and the calls it received."""

    waveform: np.ndarray
    calls: list[tuple[Path, Callable[[], None]]]


@pytest.fixture
def decoding(monkeypatch: pytest.MonkeyPatch) -> FakeDecoding:
    """Replace decoding with a fixed half-second waveform, and record each call."""
    fake = FakeDecoding(np.full(8000, 0.25, dtype=np.float32), [])

    def fake_decode(media: Path, checkpoint: Callable[[], None]) -> np.ndarray:
        fake.calls.append((media, checkpoint))
        return fake.waveform

    monkeypatch.setattr(testee, 'decode_waveform', fake_decode)
    return fake


def test_run_pipeline_returns_srt_text_counts_and_stage_timings(decoding: FakeDecoding) -> None:
    """The result carries the serialized text, the raw segment count, the audio length and each stage's time."""
    model = FakeModel(SEGMENTS)

    def checkpoint() -> None:
        pass

    result = testee.run_pipeline(
        Path('media.mp4'), lambda: model, 'en', checkpoint, stepping_clock(10.0, 11.0, 15.0, 17.0, 20.0),
    )

    assert result == testee.PipelineResult(
        srt_text=segments_to_srt(SEGMENTS),
        segment_count=3,
        audio_seconds=0.5,
        decoding_seconds=1.0,
        load_seconds=4.0,
        transcription_seconds=2.0,
        serialization_seconds=3.0,
    )
    assert decoding.calls == [(Path('media.mp4'), checkpoint)]
    [(passed_waveform, language)] = model.calls
    assert passed_waveform is decoding.waveform
    assert language == 'en'


def test_run_pipeline_consumes_every_segment_before_serializing(
    decoding: FakeDecoding, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serialization sees a fully consumed engine output, so lazy work is finished before any text exists."""
    model = FakeModel(SEGMENTS)
    consumed_at_serialization: list[int] = []

    def spy(segments: list[Segment]) -> str:
        consumed_at_serialization.append(model.consumed)
        return segments_to_srt(segments)

    monkeypatch.setattr(testee, 'segments_to_srt', spy)

    testee.run_pipeline(Path('media.mp4'), lambda: model, 'en')

    assert consumed_at_serialization == [len(SEGMENTS)]


def test_run_pipeline_rejects_the_run_before_serializing(
    decoding: FakeDecoding, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected segment ends the run with no SRT text, even when it is not the first segment."""
    model = FakeModel([Segment(0, 1, 'fine'), Segment(1, 2, REJECTED_SUBSTRINGS[0]), Segment(2, 3, 'later')])
    serialized: list[object] = []
    monkeypatch.setattr(testee, 'segments_to_srt', serialized.append)

    with pytest.raises(RejectedTranscriptionError):
        testee.run_pipeline(Path('media.mp4'), lambda: model, 'zh')

    assert serialized == []


def test_run_pipeline_propagates_an_engine_failure(decoding: FakeDecoding) -> None:
    """An error raised while the lazy segments are produced propagates unchanged."""
    model = FakeModel(SEGMENTS, error=MemoryError('device out of memory'))

    with pytest.raises(MemoryError, match='device out of memory'):
        testee.run_pipeline(Path('media.mp4'), lambda: model, 'en')


def test_run_pipeline_does_no_model_work_for_empty_audio(tmp_path: Path) -> None:
    """Real decoding of a file with no samples fails before the model is loaded or called."""
    media = tmp_path / 'empty.wav'
    with wave.open(str(media), 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
    loads: list[FakeModel] = []

    def load() -> FakeModel:
        loads.append(FakeModel(SEGMENTS))
        return loads[-1]

    with pytest.raises(EmptyAudioError):
        testee.run_pipeline(media, load, 'en')

    assert loads == []


def test_run_pipeline_end_to_end_with_real_decoding(tmp_path: Path) -> None:
    """With real ffmpeg decoding, the model receives the decoded samples and the text is complete."""
    media = tmp_path / 'clip.wav'
    with wave.open(str(media), 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(np.full(1600, 16384, dtype='<i2').tobytes())
    model = FakeModel([Segment(0.0, 0.1, 'beep')])

    result = testee.run_pipeline(media, lambda: model, 'en')

    [(passed_waveform, _)] = model.calls
    np.testing.assert_array_equal(passed_waveform, np.full(1600, 0.5, dtype=np.float32))
    assert result.audio_seconds == pytest.approx(0.1)
    assert result.srt_text == '1\n00:00:00,000 --> 00:00:00,100\nbeep\n'


def test_run_pipeline_loads_the_model_after_decoding_and_before_transcribing(decoding: FakeDecoding) -> None:
    """The loader runs once, after decoding has finished, and its model is the one that transcribes (M25)."""
    model = FakeModel(SEGMENTS)
    events: list[str] = []

    def load() -> FakeModel:
        events.append(f'load after {len(decoding.calls)} decodings, {len(model.calls)} transcriptions')
        return model

    testee.run_pipeline(Path('media.mp4'), load, 'en')

    assert events == ['load after 1 decodings, 0 transcriptions']
    assert len(model.calls) == 1


def test_run_pipeline_propagates_a_loader_failure_without_transcribing(decoding: FakeDecoding) -> None:
    """An error from the loader propagates unchanged after decoding, and nothing is transcribed."""

    def load() -> FakeModel:
        raise LookupError('no such profile')

    with pytest.raises(LookupError, match='no such profile'):
        testee.run_pipeline(Path('media.mp4'), load, 'en')

    assert len(decoding.calls) == 1
