"""The MLX Whisper engine adapter for Apple silicon (DESIGN section 8, M13, M23).

mlx-whisper is installed only on macOS, so this module imports it, and MLX, only inside the
methods that need them. The module therefore imports on Linux, where the gates type-check it
against the stubs in `stubs/` and the tests run it against fake modules.

mlx-whisper 0.4.3's `transcribe` takes no model object. It always fetches the model from the
global `ModelHolder`, which keeps one model and recognizes it by path alone, ignoring the dtype.
The adapter therefore loads through `ModelHolder` and passes `transcribe` the same directory and
precision, so that `transcribe` finds the loaded model instead of loading or downloading one.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from media_transcriber.engine import Profile, Segment, Waveform
from media_transcriber.weights import resolve_weights
from media_transcriber.whisper_languages import WHISPER_LANGUAGES, whisper_language

if TYPE_CHECKING:
    from mlx_whisper.whisper import Whisper

__all__ = ['BASELINE_PROFILE', 'ENGINE', 'MlxWhisperAdapter', 'MlxWhisperModel']

ENGINE = 'mlx-whisper'

BASELINE_PROFILE = Profile(
    name='mlx-whisper-large-v3',
    engine=ENGINE,
    # The model and revision of T2's smoke test on the M2 Mac.
    model='mlx-community/whisper-large-v3-mlx',
    revision='49e6aa286ad60c14352c404340ded53710378a11',
    precision='float16',
)
"""The Mac baseline: full `large-v3` in `float16`, mlx-whisper's default precision (M23)."""

_FP16_BY_PRECISION = {'float16': True, 'float32': False}
"""mlx-whisper's `fp16` flag for each precision that it supports."""


class MlxWhisperModel:
    """An MLX Whisper model that `ModelHolder` holds, ready for one waveform after another."""

    def __init__(self, directory: str, fp16: bool, model: Whisper) -> None:
        """Wrap the model that `ModelHolder` loaded from a snapshot directory at a precision."""
        self._directory = directory
        self._fp16 = fp16
        self._model = model

    def transcribe(self, waveform: Waveform, language: str) -> list[Segment]:
        """Transcribe a waveform with the library's default decoding options.

        mlx-whisper finishes the whole transcription before it returns, so the segments are a list.

        Args:
            waveform: The decoded audio.
            language: A Whisper language code.

        Returns:
            The segments in time order.

        Raises:
            RuntimeError: If a later `load` has replaced this model in `ModelHolder`. Without the
                check, `transcribe` would silently use the newer model, or load this one again.
        """
        from mlx_whisper.transcribe import ModelHolder, transcribe

        if ModelHolder.model is not self._model:
            raise RuntimeError(f'The MLX model from {self._directory} is no longer loaded. Load its profile again.')
        result = transcribe(
            waveform, path_or_hf_repo=self._directory, fp16=self._fp16, language=language, task='transcribe',
        )
        return [Segment(start=segment['start'], end=segment['end'], text=segment['text'])
                for segment in result['segments']]


class MlxWhisperAdapter:
    """The adapter that loads MLX Whisper profiles from the weights directory."""

    name = ENGINE

    def __init__(self, weights_dir: Path) -> None:
        """Create an adapter that loads weights from a directory laid out as a Hugging Face cache (M22)."""
        self.weights_dir = weights_dir

    def supported_languages(self) -> frozenset[str]:
        """Return the Whisper language codes."""
        return WHISPER_LANGUAGES

    def engine_language(self, tag: str) -> str | None:
        """Map a BCP 47 tag to its Whisper code, or None if Whisper does not support it."""
        return whisper_language(tag)

    def load(self, profile: Profile) -> MlxWhisperModel:
        """Load a profile's model from the weights directory into `ModelHolder`, without any download.

        The previous model in `ModelHolder` is dropped first, even if it came from the same
        directory, because `ModelHolder` would otherwise return it whatever its precision. A model
        that an earlier call returned then refuses to transcribe. Its memory is freed once the
        caller drops it too.

        Args:
            profile: An MLX Whisper profile in `float16` or `float32`, without decoding settings.

        Returns:
            The loaded model.

        Raises:
            ValueError: If the profile is for another engine, has decoding settings, has another
                precision, or names an invalid repository.
            WeightsNotFoundError: If the weights directory lacks the profile's snapshot. The model
                that `ModelHolder` holds is then kept.
        """
        if profile.engine != ENGINE:
            raise ValueError(f'Profile {profile.name} is for engine {profile.engine}, not {ENGINE}.')
        if profile.decoding:
            raise ValueError(f'Engine {ENGINE} accepts no decoding settings, but profile {profile.name} has '
                             f'{sorted(profile.decoding)}.')
        if profile.precision not in _FP16_BY_PRECISION:
            raise ValueError(f'Engine {ENGINE} supports the precisions {sorted(_FP16_BY_PRECISION)}, but profile '
                             f'{profile.name} has {profile.precision}.')
        fp16 = _FP16_BY_PRECISION[profile.precision]
        directory = str(resolve_weights(self.weights_dir, profile))

        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        ModelHolder.model = None
        ModelHolder.model_path = None
        model = ModelHolder.get_model(directory, mx.float16 if fp16 else mx.float32)
        return MlxWhisperModel(directory, fp16, model)
