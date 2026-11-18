"""Tests for decoding media into the waveform, with ffmpeg on synthetic audio only."""

import io
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

import media_transcriber.audio as testee


def write_wav(path: Path, samples: list[int], rate: int = testee.SAMPLE_RATE) -> Path:
    """Write mono 16-bit samples as a WAV file, so decoding needs no resampling."""
    with wave.open(str(path), 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(np.array(samples, dtype='<i2').tobytes())
    return path


def synthesize(path: Path, *source: str) -> Path:
    """Render a synthetic lavfi source into `path` with ffmpeg; the test owns the result."""
    subprocess.run(
        ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', *source, str(path)],
        check=True,
    )
    return path


class RecordingPopen(subprocess.Popen[bytes]):
    """A Popen that remembers each child it starts, so a test can check that it was reaped."""

    started: list[RecordingPopen] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Start the child and record it."""
        super().__init__(*args, **kwargs)
        if 'pipe:1' in self.args:  # the decoder's command, not a test's own synthesis
            RecordingPopen.started.append(self)


@pytest.fixture
def children(monkeypatch: pytest.MonkeyPatch) -> list[RecordingPopen]:
    """Record every ffmpeg child that the decoder starts during the test."""
    started: list[RecordingPopen] = []
    monkeypatch.setattr(RecordingPopen, 'started', started)
    monkeypatch.setattr(testee.subprocess, 'Popen', RecordingPopen)
    return started


def assert_reaped(children: list[RecordingPopen]) -> None:
    """Assert that the decoder's one child was reaped, its pipes closed and its drain thread joined."""
    [child] = children
    assert child.returncode is not None
    assert child.stdout is not None and child.stdout.closed
    assert child.stderr is not None and child.stderr.closed
    assert not [thread for thread in threading.enumerate() if thread.name == 'ffmpeg-stderr']


def test_decode_waveform_scales_16_bit_samples_exactly(tmp_path: Path, children: list[RecordingPopen]) -> None:
    """Mono 16 kHz PCM passes through unchanged and is scaled by 1/32768 into float32."""
    samples = [0, 1, -1, 16384, -32768, 32767]

    waveform = testee.decode_waveform(write_wav(tmp_path / 'clip.wav', samples))

    assert waveform.dtype == np.float32
    np.testing.assert_array_equal(waveform, np.array(samples, dtype=np.float32) / 32768)
    assert_reaped(children)


def test_decode_waveform_downmixes_and_resamples(tmp_path: Path) -> None:
    """A stereo 44.1 kHz source becomes mono at 16 kHz, so one second gives 16000 samples."""
    clip = synthesize(tmp_path / 'tone.mka', '-i', 'sine=frequency=440:sample_rate=44100:duration=1', '-ac', '2')

    waveform = testee.decode_waveform(clip)

    # The resampler's edges shift the count by a few hundred samples. 5% still tells 16 kHz mono
    # apart from 44.1 kHz or from interleaved stereo, which would give 44100 or 32000 samples.
    assert abs(waveform.size - testee.SAMPLE_RATE) <= testee.SAMPLE_RATE // 20
    assert 0.05 < float(np.abs(waveform).max()) < 1


def test_decode_waveform_reads_a_name_that_looks_like_a_protocol(tmp_path: Path) -> None:
    """A file named like an ffmpeg protocol is opened as a file, not as that protocol."""
    clip = write_wav(tmp_path / 'concat:clip.wav', [5, 6, 7])

    assert testee.decode_waveform(clip).size == 3


def test_decode_waveform_rejects_audio_without_samples(tmp_path: Path, children: list[RecordingPopen]) -> None:
    """A valid file with no samples raises EmptyAudioError instead of returning an empty waveform."""
    with pytest.raises(testee.EmptyAudioError, match='No audio samples'):
        testee.decode_waveform(write_wav(tmp_path / 'empty.wav', []))

    assert_reaped(children)


@pytest.mark.parametrize('kind', ['garbage', 'video-only'])
def test_decode_waveform_reports_ffmpeg_failures(tmp_path: Path, children: list[RecordingPopen], kind: str) -> None:
    """A file that ffmpeg cannot decode to audio raises DecodingError with the status and diagnostics."""
    media = tmp_path / 'input.mkv'
    if kind == 'garbage':
        media.write_bytes(b'not media at all')
    else:
        synthesize(media, '-i', 'testsrc=size=32x32:duration=1', '-c:v', 'ffv1')

    with pytest.raises(testee.DecodingError) as raised:
        testee.decode_waveform(media)

    assert raised.value.returncode != 0
    assert raised.value.stderr.strip()
    assert str(media) in str(raised.value)
    assert_reaped(children)


def test_decode_waveform_calls_the_checkpoint_after_each_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The checkpoint runs once per chunk of PCM, so the worker can check the marker while decoding."""
    monkeypatch.setattr(testee, 'CHUNK_BYTES', 1000)
    clip = write_wav(tmp_path / 'clip.wav', [0] * 2500)
    calls: list[int] = []

    testee.decode_waveform(clip, lambda: calls.append(1))

    # 5000 bytes of PCM arrive in at least five reads of at most 1000 bytes.
    assert len(calls) >= 5


def test_decode_waveform_kills_and_reaps_ffmpeg_when_the_checkpoint_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, children: list[RecordingPopen],
) -> None:
    """An exception from the checkpoint stops ffmpeg mid-stream, reaps it and propagates unchanged."""
    monkeypatch.setattr(testee, 'CHUNK_BYTES', 1000)
    # Ten seconds of PCM is 320 kB, far beyond a pipe's buffer, so ffmpeg is still writing when
    # the first checkpoint raises and only a kill ends it.
    clip = write_wav(tmp_path / 'long.wav', [0] * (10 * testee.SAMPLE_RATE))

    class Stop(Exception):
        """The checkpoint's deliberate failure."""

    def checkpoint() -> None:
        raise Stop

    with pytest.raises(Stop):
        testee.decode_waveform(clip, checkpoint)

    assert_reaped(children)
    assert children[0].returncode < 0  # ended by a signal, not by finishing


def test_decode_waveform_reaps_ffmpeg_when_the_drain_thread_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, children: list[RecordingPopen],
) -> None:
    """If the drain thread fails to start, ffmpeg is still killed and reaped before the error propagates."""

    class UnstartableThread(threading.Thread):
        def start(self) -> None:
            raise RuntimeError('cannot start thread')

    monkeypatch.setattr(testee.threading, 'Thread', UnstartableThread)

    with pytest.raises(RuntimeError, match='cannot start thread'):
        testee.decode_waveform(write_wav(tmp_path / 'clip.wav', [0] * (10 * testee.SAMPLE_RATE)))

    [child] = children
    assert child.returncode is not None
    assert child.stdout is not None and child.stdout.closed


def test_kill_decoders_ends_a_running_decoder_and_forgets_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, children: list[RecordingPopen],
) -> None:
    """A shutdown's kill reaches the running ffmpeg child, which is still reaped by its own call (DESIGN 5.9)."""
    monkeypatch.setattr(testee, '_accepting', True)  # the kill closes the registry; restored after the test
    monkeypatch.setattr(testee, 'CHUNK_BYTES', 1000)
    clip = write_wav(tmp_path / 'long.wav', [0] * (10 * testee.SAMPLE_RATE))
    kills: list[int] = []

    def checkpoint() -> None:
        if not kills:
            kills.append(testee.kill_decoders())

    with pytest.raises(testee.DecodingError) as raised:
        testee.decode_waveform(clip, checkpoint)

    assert kills == [1]
    assert raised.value.returncode < 0  # ended by the kill, not by finishing
    assert_reaped(children)
    assert testee.kill_decoders() == 0  # a finished decoder is no longer tracked


def test_no_decoder_starts_after_kill_decoders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, children: list[RecordingPopen],
) -> None:
    """A worker that resumes after the shutdown's kill cannot start a decoder that would outlive the process."""
    monkeypatch.setattr(testee, '_accepting', True)  # restored after the test
    assert testee.kill_decoders() == 0

    with pytest.raises(testee.DecodingStoppedError):
        testee.decode_waveform(write_wav(tmp_path / 'clip.wav', [0] * 100))

    assert children == []


def test_drain_keeps_only_the_tail_of_the_stream() -> None:
    """The drain reads to the end but keeps only the last STDERR_TAIL_BYTES bytes."""
    data = bytes(range(256)) * (3 * testee.STDERR_TAIL_BYTES // 256 + 1)
    tail = bytearray()

    testee._drain(io.BytesIO(data), tail)

    assert bytes(tail) == data[-testee.STDERR_TAIL_BYTES:]
