"""A scripted engine model that stands in for real engines in tests (DESIGN section 8, M10).

The fake model yields its scripted segments lazily, as real engines do, and counts how many were
consumed. A test can therefore tell whether the code under test read every segment, or stopped
early, without a real model.
"""

from collections.abc import Iterable, Iterator, Sequence

from media_transcriber.engine import Segment, Waveform


class FakeModel:
    """A loaded model that yields scripted segments and records its calls."""

    def __init__(self, segments: Sequence[Segment] = (), error: BaseException | None = None) -> None:
        """Script the segments to yield, and an optional error to raise after the last one."""
        self.segments = list(segments)
        self.error = error
        self.calls: list[tuple[Waveform, str]] = []
        self.consumed = 0

    def transcribe(self, waveform: Waveform, language: str) -> Iterable[Segment]:
        """Record the call and return a lazy iterator over the scripted segments."""
        self.calls.append((waveform, language))
        return self._generate()

    def _generate(self) -> Iterator[Segment]:
        """Yield each scripted segment, counting it as consumed, then raise the scripted error."""
        for segment in self.segments:
            self.consumed += 1
            yield segment
        if self.error is not None:
            raise self.error

