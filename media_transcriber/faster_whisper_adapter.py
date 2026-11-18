"""The faster-whisper engine adapter for NVIDIA CUDA (DESIGN section 8, M13).

The adapter reproduces the `youtube-whisperer` source revision's settings: a CUDA model in
`float32`, transcribed sequentially with the library's default decoding options. Batched
transcription is a different profile and is not supported yet.
"""

from collections.abc import Iterable, Iterator
from pathlib import Path

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment as LibrarySegment

from media_transcriber.engine import Profile, Segment, Waveform
from media_transcriber.weights import WeightsNotFoundError, resolve_weights
from media_transcriber.whisper_languages import WHISPER_LANGUAGES, whisper_language

__all__ = ['BASELINE_PROFILE', 'ENGINE', 'FasterWhisperAdapter', 'FasterWhisperModel']

ENGINE = 'faster-whisper'

BASELINE_PROFILE = Profile(
    name='faster-whisper-large-v3',
    engine=ENGINE,
    # faster-whisper maps the source revision's `large-v3` to this repository. The revision was
    # its `main` on 2026-11-17.
    model='Systran/faster-whisper-large-v3',
    revision='edaa852ec7e145841d8ffdb056a99866b5f0a478',
    precision='float32',
)
"""The CUDA baseline: full `large-v3` in `float32`, sequential, as at the source revision."""


class FasterWhisperModel:
    """A faster-whisper model loaded on the GPU, ready for one waveform after another."""

    def __init__(self, model: WhisperModel) -> None:
        """Wrap a loaded faster-whisper model."""
        self._model = model

    def transcribe(self, waveform: Waveform, language: str) -> Iterable[Segment]:
        """Transcribe a waveform with the library's default decoding options.

        faster-whisper prepares the audio at once, but decodes each segment only when the caller
        consumes it. The returned iterator keeps that laziness, so the work is finished only when
        the caller has consumed every segment.

        Args:
            waveform: The decoded audio.
            language: A Whisper language code.

        Returns:
            A lazy iterator over the segments in time order.
        """
        segments, _info = self._model.transcribe(waveform, language=language, task='transcribe')
        return _to_segments(segments)


def _to_segments(segments: Iterable[LibrarySegment]) -> Iterator[Segment]:
    """Convert faster-whisper's segments to engine-neutral ones, one at a time as they are consumed."""
    for segment in segments:
        yield Segment(start=segment.start, end=segment.end, text=segment.text)


class FasterWhisperAdapter:
    """The adapter that loads faster-whisper profiles from the weights directory."""

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

    def accelerator_available(self) -> bool:
        """Return whether CTranslate2 sees at least one CUDA device."""
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0

    def load(self, profile: Profile) -> FasterWhisperModel:
        """Load a profile's model on the GPU from the weights directory, without any download.

        Args:
            profile: A faster-whisper profile without decoding settings.

        Returns:
            The loaded model.

        Raises:
            ValueError: If the profile is for another engine, has decoding settings, or names an
                invalid repository. CTranslate2 also raises it for an unknown precision.
            WeightsNotFoundError: If the weights directory lacks the profile's snapshot, or the
                snapshot lacks `tokenizer.json`.
            RuntimeError: If CTranslate2 cannot use CUDA.
        """
        if profile.engine != ENGINE:
            raise ValueError(f'Profile {profile.name} is for engine {profile.engine}, not {ENGINE}.')
        if profile.decoding:
            raise ValueError(f'Engine {ENGINE} accepts no decoding settings, but profile {profile.name} has '
                             f'{sorted(profile.decoding)}.')
        directory = resolve_weights(self.weights_dir, profile)
        # faster-whisper 1.2.1 replaces a missing tokenizer.json with openai/whisper-tiny's from the
        # Hub, and does not pass local_files_only to that lookup. Its other files fail or fall
        # back locally.
        if not (directory / 'tokenizer.json').is_file():
            raise WeightsNotFoundError(f'The weights of {profile.model} at revision {profile.revision} in '
                                       f'{directory} lack tokenizer.json. Download the complete snapshot.')
        # faster-whisper treats a name that is not an existing directory as a repository to
        # download. If the snapshot disappears before this call, local_files_only still keeps
        # the library from downloading (DESIGN section 9).
        model = WhisperModel(str(directory), device='cuda', compute_type=profile.precision, local_files_only=True)
        return FasterWhisperModel(model)
