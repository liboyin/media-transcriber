"""Tests for the faster-whisper adapter, with an autospec fake of the installed `WhisperModel`.

The fake takes the real class's signatures, so a call that faster-whisper would reject fails here
too. The constructor also takes `**model_kwargs`, which accepts any name, so the tests assert its
exact arguments. The fake yields the library's own `Segment` dataclass, so the adapter reads the
real field names.
"""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, Mock, create_autospec

import faster_whisper.transcribe as library
import numpy as np
import pytest
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment as LibrarySegment
from hub_cache import add_snapshot

import media_transcriber.faster_whisper_adapter as testee
from media_transcriber.engine import Segment
from media_transcriber.weights import WeightsNotFoundError

SNAPSHOT_FILES = ('config.json', 'model.bin', 'tokenizer.json')


def add_baseline_snapshot(weights_dir: Path) -> Path:
    """Add a complete snapshot of the baseline profile to a test-owned weights directory, and return it."""
    return add_snapshot(weights_dir, testee.BASELINE_PROFILE.model, testee.BASELINE_PROFILE.revision, SNAPSHOT_FILES)


def library_segment(start: float, end: float, text: str) -> LibrarySegment:
    """Build a faster-whisper segment with the given timing and text."""
    return LibrarySegment(
        id=0, seek=0, start=start, end=end, text=text, tokens=[], avg_logprob=0.0, compression_ratio=1.0,
        no_speech_prob=0.0, words=None, temperature=0.0,
    )


@pytest.fixture
def whisper_model(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the adapter's `WhisperModel` with an autospec fake of the installed class, and return it."""
    fake = create_autospec(WhisperModel)
    monkeypatch.setattr(testee, 'WhisperModel', fake)
    return fake


def test_baseline_profile_is_full_large_v3_in_float32_without_decoding_settings() -> None:
    """The CUDA baseline is the source revision's model and precision, pinned to a commit."""
    profile = testee.BASELINE_PROFILE

    assert profile.engine == testee.ENGINE == 'faster-whisper'
    assert profile.model == 'Systran/faster-whisper-large-v3'
    assert len(profile.revision) == 40
    assert profile.precision == 'float32'
    assert profile.decoding == {}


def test_adapter_reports_whisper_languages_and_maps_tags(tmp_path: Path) -> None:
    """The adapter offers Whisper's languages and maps a tag to its primary subtag."""
    adapter = testee.FasterWhisperAdapter(tmp_path)

    assert adapter.name == 'faster-whisper'
    assert 'zh' in adapter.supported_languages()
    assert adapter.engine_language('zh-TW') == 'zh'
    assert adapter.engine_language('tlh') is None


def test_load_builds_a_cuda_model_from_the_resolved_snapshot_offline(tmp_path: Path, whisper_model: MagicMock) -> None:
    """The library gets the local snapshot directory, CUDA, the profile's precision and no Hub access."""
    profile = testee.BASELINE_PROFILE
    snapshot = add_baseline_snapshot(tmp_path)

    model = testee.FasterWhisperAdapter(tmp_path).load(profile)

    whisper_model.assert_called_once_with(str(snapshot), device='cuda', compute_type='float32', local_files_only=True)
    assert isinstance(model, testee.FasterWhisperModel)


def test_load_rejects_a_profile_for_another_engine_before_touching_the_weights(
    tmp_path: Path, whisper_model: MagicMock,
) -> None:
    """A profile for another engine fails before the lookup, so even a missing directory is not read."""
    profile = replace(testee.BASELINE_PROFILE, engine='mlx-whisper')

    with pytest.raises(ValueError, match='mlx-whisper'):
        testee.FasterWhisperAdapter(tmp_path / 'missing').load(profile)

    whisper_model.assert_not_called()


def test_load_rejects_decoding_settings(tmp_path: Path, whisper_model: MagicMock) -> None:
    """The adapter accepts no decoding settings yet, so a profile with one fails instead of being ignored."""
    profile = replace(testee.BASELINE_PROFILE, decoding={'beam_size': 1})
    add_baseline_snapshot(tmp_path)

    with pytest.raises(ValueError, match='beam_size'):
        testee.FasterWhisperAdapter(tmp_path).load(profile)

    whisper_model.assert_not_called()


def test_load_fails_without_building_a_model_when_the_weights_are_missing(
    tmp_path: Path, whisper_model: MagicMock,
) -> None:
    """Missing weights stop the load before the library could fall back to a download."""
    with pytest.raises(WeightsNotFoundError):
        testee.FasterWhisperAdapter(tmp_path).load(testee.BASELINE_PROFILE)

    whisper_model.assert_not_called()


def test_load_refuses_a_snapshot_without_a_tokenizer_before_the_library_could_fetch_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without tokenizer.json, faster-whisper would fetch a tokenizer from the Hub, so the load stops first.

    The library's real Python constructor runs, with native model loading and the remote tokenizer
    lookup replaced by doubles, so a removed check shows up as a recorded remote call.
    """
    snapshot = add_snapshot(tmp_path, testee.BASELINE_PROFILE.model, testee.BASELINE_PROFILE.revision,
                            ('config.json', 'model.bin'))
    native = Mock(return_value=Mock(is_multilingual=True))
    monkeypatch.setattr(library.ctranslate2.models, 'Whisper', native)
    remote = Mock(side_effect=RuntimeError('remote tokenizer lookup intercepted'))
    monkeypatch.setattr(library.tokenizers.Tokenizer, 'from_pretrained', remote)

    with pytest.raises(WeightsNotFoundError, match='tokenizer.json') as caught:
        testee.FasterWhisperAdapter(tmp_path).load(testee.BASELINE_PROFILE)

    assert str(snapshot) in str(caught.value)
    remote.assert_not_called()
    native.assert_not_called()


def test_transcribe_passes_the_language_and_maps_segments_lazily(tmp_path: Path, whisper_model: MagicMock) -> None:
    """The library gets the waveform, the code and the task, and each segment converts only when consumed."""
    produced: list[LibrarySegment] = []
    scripted = [library_segment(0.0, 1.5, ' hello'), library_segment(1.5, 3.0, 'world ')]

    def library_segments() -> Iterator[LibrarySegment]:
        for segment in scripted:
            produced.append(segment)
            yield segment

    instance = whisper_model.return_value
    instance.transcribe.return_value = (library_segments(), object())
    add_baseline_snapshot(tmp_path)
    model = testee.FasterWhisperAdapter(tmp_path).load(testee.BASELINE_PROFILE)
    waveform = np.zeros(16000, dtype=np.float32)

    segments = iter(model.transcribe(waveform, 'zh'))

    instance.transcribe.assert_called_once_with(waveform, language='zh', task='transcribe')
    assert produced == []
    assert next(segments) == Segment(0.0, 1.5, ' hello')
    assert len(produced) == 1
    assert list(segments) == [Segment(1.5, 3.0, 'world ')]
