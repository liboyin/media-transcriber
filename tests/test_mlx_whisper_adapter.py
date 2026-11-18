"""Tests for the MLX Whisper adapter, with fake `mlx` and `mlx_whisper` modules.

MLX is installed only on macOS. The fakes keep the behavior of mlx-whisper 0.4.3 that the adapter
relies on: `ModelHolder` keeps one model and recognizes it by path alone, and `transcribe` fetches
its model from `ModelHolder`. A fake that reloaded on every precision change would hide the
adapter's reason to drop the held model.
"""

import subprocess
import sys
import types
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from hub_cache import add_snapshot

import media_transcriber.mlx_whisper_adapter as testee
from media_transcriber.engine import Segment
from media_transcriber.weights import WeightsNotFoundError


@dataclass(frozen=True)
class FakeDtype:
    """Stands in for an `mlx.core.Dtype`."""

    name: str


FLOAT16 = FakeDtype('float16')
FLOAT32 = FakeDtype('float32')


@dataclass(eq=False)
class FakeWhisper:
    """Stands in for a loaded `mlx_whisper.whisper.Whisper`, and remembers how it was loaded."""

    path: str
    dtype: FakeDtype


@dataclass
class FakeMlx:
    """The fake modules' state: the holder, the scripted segments, and the calls they received."""

    holder: type
    segments: list[dict[str, object]] = field(default_factory=list)
    loads: list[tuple[str, FakeDtype]] = field(default_factory=list)
    metal: bool = True
    transcriptions: list[dict[str, object]] = field(default_factory=list)


@pytest.fixture
def fake_mlx(monkeypatch: pytest.MonkeyPatch) -> FakeMlx:
    """Install fake `mlx.core` and `mlx_whisper.transcribe` modules for one test, and return their state."""

    class FakeModelHolder:
        """Keeps one model, recognized by path alone, as mlx-whisper 0.4.3 does."""

        model: ClassVar[FakeWhisper | None] = None
        model_path: ClassVar[str | None] = None

        @classmethod
        def get_model(cls, model_path: str, dtype: FakeDtype) -> FakeWhisper:
            if cls.model is None or model_path != cls.model_path:
                state.loads.append((model_path, dtype))
                cls.model = FakeWhisper(model_path, dtype)
                cls.model_path = model_path
            return cls.model

    def fake_transcribe(audio: np.ndarray, **options: object) -> dict[str, object]:
        state.transcriptions.append({'audio': audio, **options})
        FakeModelHolder.get_model(options['path_or_hf_repo'], FLOAT16 if options.get('fp16', True) else FLOAT32)
        return {'text': 'ignored', 'segments': state.segments, 'language': options.get('language')}

    state = FakeMlx(FakeModelHolder)
    core = types.ModuleType('mlx.core')
    metal = types.ModuleType('mlx.core.metal')
    vars(metal).update(is_available=lambda: state.metal)
    vars(core).update(float16=FLOAT16, float32=FLOAT32, metal=metal)
    mlx = types.ModuleType('mlx')
    vars(mlx).update(core=core)
    transcribe_module = types.ModuleType('mlx_whisper.transcribe')
    vars(transcribe_module).update(ModelHolder=FakeModelHolder, transcribe=fake_transcribe)
    package = types.ModuleType('mlx_whisper')
    vars(package).update(transcribe=fake_transcribe)
    for name, module in [('mlx', mlx), ('mlx.core', core), ('mlx_whisper', package),
                         ('mlx_whisper.transcribe', transcribe_module)]:
        monkeypatch.setitem(sys.modules, name, module)
    return state


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    """Add the baseline profile's snapshot to a test-owned weights directory, and return the snapshot."""
    return add_snapshot(tmp_path, testee.BASELINE_PROFILE.model, testee.BASELINE_PROFILE.revision)


def test_module_imports_and_maps_languages_without_mlx(tmp_path: Path) -> None:
    """On a host without MLX, the module imports and the adapter answers without loading anything."""
    code = (
        'import sys\n'
        "sys.modules.update({'mlx': None, 'mlx.core': None, 'mlx_whisper': None, 'mlx_whisper.transcribe': None})\n"
        'from pathlib import Path\n'
        'from media_transcriber.mlx_whisper_adapter import MlxWhisperAdapter\n'
        "adapter = MlxWhisperAdapter(Path('unused'))\n"
        "assert adapter.engine_language('zh-CN') == 'zh'\n"
        "assert 'en' in adapter.supported_languages()\n"
    )

    subprocess.run([sys.executable, '-c', code], cwd=tmp_path, check=True, timeout=60)


def test_baseline_profile_is_the_smoke_tested_large_v3_in_float16() -> None:
    """The Mac baseline is full large-v3 at the smoke-tested revision, in float16 (DESIGN M23)."""
    profile = testee.BASELINE_PROFILE

    assert profile.engine == testee.ENGINE == 'mlx-whisper'
    assert profile.model == 'mlx-community/whisper-large-v3-mlx'
    assert profile.revision == '49e6aa286ad60c14352c404340ded53710378a11'
    assert profile.precision == 'float16'
    assert profile.decoding == {}


def test_adapter_reports_whisper_languages_and_maps_tags(tmp_path: Path) -> None:
    """The adapter offers Whisper's languages and maps a tag to its primary subtag."""
    adapter = testee.MlxWhisperAdapter(tmp_path)

    assert adapter.name == 'mlx-whisper'
    assert 'yue' in adapter.supported_languages()
    assert adapter.engine_language('en-GB') == 'en'
    assert adapter.engine_language('jv') is None


@pytest.mark.parametrize(('precision', 'dtype'), [('float16', FLOAT16), ('float32', FLOAT32)])
def test_load_loads_the_resolved_snapshot_at_the_profile_precision(
    tmp_path: Path, snapshot: Path, fake_mlx: FakeMlx, precision: str, dtype: FakeDtype,
) -> None:
    """`ModelHolder` gets the local snapshot directory, never the repository name, and the profile's dtype."""
    profile = replace(testee.BASELINE_PROFILE, precision=precision)

    testee.MlxWhisperAdapter(tmp_path).load(profile)

    assert fake_mlx.loads == [(str(snapshot), dtype)]


def test_load_reloads_the_same_directory_at_a_changed_precision(
    tmp_path: Path, snapshot: Path, fake_mlx: FakeMlx,
) -> None:
    """A precision change loads a new model, although `ModelHolder` recognizes models by path alone."""
    adapter = testee.MlxWhisperAdapter(tmp_path)

    adapter.load(testee.BASELINE_PROFILE)
    adapter.load(replace(testee.BASELINE_PROFILE, precision='float32'))

    assert fake_mlx.loads == [(str(snapshot), FLOAT16), (str(snapshot), FLOAT32)]
    assert fake_mlx.holder.model.dtype == FLOAT32


@pytest.mark.parametrize(('change', 'message'), [
    ({'engine': 'faster-whisper'}, 'faster-whisper'),
    ({'decoding': {'temperature': 0.0}}, 'temperature'),
    ({'precision': 'int8'}, 'int8'),
])
def test_load_rejects_an_unsupported_profile_before_touching_the_weights(
    tmp_path: Path, fake_mlx: FakeMlx, change: dict[str, object], message: str,
) -> None:
    """Another engine, a decoding setting or another precision fails before the lookup and any load."""
    profile = replace(testee.BASELINE_PROFILE, **change)

    with pytest.raises(ValueError, match=message):
        testee.MlxWhisperAdapter(tmp_path / 'missing').load(profile)

    assert fake_mlx.loads == []


def test_load_keeps_the_held_model_when_the_weights_are_missing(
    tmp_path: Path, snapshot: Path, fake_mlx: FakeMlx,
) -> None:
    """Missing weights fail the load before the held model is dropped, so the resident model still works."""
    adapter = testee.MlxWhisperAdapter(tmp_path)
    resident = adapter.load(testee.BASELINE_PROFILE)

    with pytest.raises(WeightsNotFoundError):
        adapter.load(replace(testee.BASELINE_PROFILE, revision='f' * 40))

    assert resident.transcribe(np.zeros(16000, dtype=np.float32), 'en') == []
    assert fake_mlx.loads == [(str(snapshot), FLOAT16)]


def test_transcribe_uses_the_loaded_model_and_maps_segments(tmp_path: Path, snapshot: Path, fake_mlx: FakeMlx) -> None:
    """`transcribe` gets the loaded model's directory and precision, so it reuses the model without a reload."""
    fake_mlx.segments = [
        {'seek': 0, 'start': 0.0, 'end': 1.5, 'text': ' hello', 'tokens': [1]},
        {'seek': 0, 'start': 1.5, 'end': 3.0, 'text': 'world ', 'tokens': [2]},
    ]
    model = testee.MlxWhisperAdapter(tmp_path).load(replace(testee.BASELINE_PROFILE, precision='float32'))
    waveform = np.zeros(16000, dtype=np.float32)

    segments = model.transcribe(waveform, 'zh')

    assert segments == [Segment(0.0, 1.5, ' hello'), Segment(1.5, 3.0, 'world ')]
    assert fake_mlx.transcriptions == [
        {'audio': waveform, 'path_or_hf_repo': str(snapshot), 'fp16': False, 'language': 'zh', 'task': 'transcribe'},
    ]
    assert fake_mlx.loads == [(str(snapshot), FLOAT32)]


def test_transcribe_refuses_a_model_that_a_later_load_replaced(
    tmp_path: Path, snapshot: Path, fake_mlx: FakeMlx,
) -> None:
    """A model object from an earlier load never transcribes with the model that replaced it."""
    adapter = testee.MlxWhisperAdapter(tmp_path)
    earlier = adapter.load(testee.BASELINE_PROFILE)
    adapter.load(replace(testee.BASELINE_PROFILE, precision='float32'))

    with pytest.raises(RuntimeError, match='no longer loaded'):
        earlier.transcribe(np.zeros(16000, dtype=np.float32), 'en')

    assert fake_mlx.transcriptions == []


@pytest.mark.parametrize('available', [True, False])
def test_accelerator_available_reports_metal(tmp_path: Path, fake_mlx: FakeMlx, available: bool) -> None:
    """The accelerator check asks MLX whether Metal is available, without loading a model."""
    fake_mlx.metal = available
    assert testee.MlxWhisperAdapter(tmp_path).accelerator_available() is available
    assert fake_mlx.loads == []


def test_accelerator_available_is_false_without_mlx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Where MLX is not installed, as on Linux, the accelerator is reported unavailable instead of raising."""
    monkeypatch.setitem(sys.modules, 'mlx', None)
    monkeypatch.setitem(sys.modules, 'mlx.core', None)
    assert testee.MlxWhisperAdapter(tmp_path).accelerator_available() is False
