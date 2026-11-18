"""A scripted engine model and adapter that stand in for real engines in tests (DESIGN section 8, M10).

The fake model yields its scripted segments lazily, as real engines do, and counts how many were
consumed. A test can therefore tell whether the code under test read every segment, or stopped
early, without a real model.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence

from media_transcriber.engine import Profile, Segment, Waveform
from media_transcriber.whisper_languages import WHISPER_LANGUAGES, whisper_language


class FakeModel:
    """A loaded model that yields scripted segments and records its calls."""

    def __init__(self, segments: Sequence[Segment] = (), error: BaseException | None = None) -> None:
        """Script the segments to yield, and an optional error to raise after the last one."""
        self.segments = list(segments)
        self.error = error
        self.calls: list[tuple[Waveform, str]] = []
        self.consumed = 0
        self.on_transcribe: list[Callable[[], None]] = []

    def transcribe(self, waveform: Waveform, language: str) -> Iterable[Segment]:
        """Record the call, run the scripted hooks, and return a lazy iterator over the scripted segments."""
        self.calls.append((waveform, language))
        for hook in self.on_transcribe:
            hook()
        return self._generate()

    def _generate(self) -> Iterator[Segment]:
        """Yield each scripted segment, counting it as consumed, then raise the scripted error."""
        for segment in self.segments:
            self.consumed += 1
            yield segment
        if self.error is not None:
            raise self.error



class FakeAdapter:
    """An engine adapter that loads scripted models and records each load.

    It supports Whisper's language codes through the real mapping, so tests exercise the same tags
    that real adapters accept.
    """

    def __init__(self, name: str = 'fake', model: FakeModel | None = None, error: BaseException | None = None) -> None:
        """Script the model that every load returns, or an error that every load raises."""
        self.name = name
        self.model = model if model is not None else FakeModel()
        self.error = error
        self.loads: list[str] = []
        self.on_load: list[Callable[[], None]] = []
        self.accelerator = True

    def supported_languages(self) -> frozenset[str]:
        """Return Whisper's language codes."""
        return WHISPER_LANGUAGES

    def engine_language(self, tag: str) -> str | None:
        """Map a tag as the Whisper adapters do."""
        return whisper_language(tag)

    def load(self, profile: Profile) -> FakeModel:
        """Record the load, run the scripted hooks, then raise the scripted error or return the model."""
        self.loads.append(profile.name)
        for hook in self.on_load:
            hook()
        if self.error is not None:
            raise self.error
        return self.model

    def accelerator_available(self) -> bool:
        """Report the scripted availability."""
        return self.accelerator
