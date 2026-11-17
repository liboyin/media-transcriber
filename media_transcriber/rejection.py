"""Reject a whole transcription when the engine emits a known bad phrase (DESIGN 5.5 step 7).

Whisper sometimes hallucinates credits from its training data on silence or music. A run that
contains one is untrustworthy as a whole, so it is rejected rather than filtered. The phrases are
those of `youtube-whisperer`'s `rejection_policy.py` at the source revision, and they apply to
every engine.
"""

from collections.abc import Iterable, Iterator

from media_transcriber.engine import Segment

__all__ = ['REJECTED_SUBSTRINGS', 'RejectedTranscriptionError', 'reject_bad_segments']

REJECTED_SUBSTRINGS: tuple[str, ...] = (
    '明镜与点点',
    '杨茜茜',
)


class RejectedTranscriptionError(RuntimeError):
    """A segment contained a rejected substring, so the whole run is rejected.

    Attributes:
        substring: The rejected substring that was found.
    """

    def __init__(self, substring: str) -> None:
        """Name the substring that caused the rejection."""
        super().__init__(f'Segment contains rejected substring: {substring!r}')
        self.substring = substring


def reject_bad_segments(segments: Iterable[Segment]) -> Iterator[Segment]:
    """Re-yield segments, and raise at the first one that contains a rejected substring.

    The check runs as the segments are consumed, so a lazy engine output is not read ahead.

    Args:
        segments: The engine's segments.

    Yields:
        Each segment that contains no rejected substring.

    Raises:
        RejectedTranscriptionError: At the first segment that contains a rejected substring.
    """
    for segment in segments:
        for substring in REJECTED_SUBSTRINGS:
            if substring in segment.text:
                raise RejectedTranscriptionError(substring)
        yield segment
