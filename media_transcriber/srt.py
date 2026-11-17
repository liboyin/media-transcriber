"""Serialize engine-neutral segments as SRT text (DESIGN 5.6, M19).

The rules are those of `youtube-whisperer`'s shared formatter in `adaptors/srt_deduplicator.py` at
the source revision: millisecond rounding before decomposition, hours always present, a comma
before the milliseconds, and outer whitespace stripped from the text. Consecutive segments with
identical text merge into one cue, as the source revision's transcription path did.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from media_transcriber.engine import Segment

__all__ = ['format_timestamp', 'segments_to_srt']


@dataclass
class _Cue:
    """One SRT cue before numbering: its times and its text lines."""

    start: float
    end: float
    lines: list[str]


def format_timestamp(seconds: float) -> str:
    """Format a time as `HH:MM:SS,mmm`, with at least two hour digits and no upper bound.

    The total is rounded to whole milliseconds before it is split, so a carry such as 59.9996
    seconds becomes `00:01:00,000` rather than an invalid `00:00:59,1000`.

    Args:
        seconds: The time in seconds.

    Returns:
        The SRT timestamp.

    Raises:
        ValueError: If `seconds` is negative.
    """
    if seconds < 0:
        raise ValueError(f'An SRT timestamp cannot be negative: {seconds}')
    milliseconds = round(seconds * 1000.0)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1_000)
    return f'{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}'


def _merged_cues(segments: Iterable[Segment]) -> Iterator[_Cue]:
    """Turn segments into cues, merging each run of consecutive cues with identical lines (M19).

    A merged cue keeps the first segment's start and takes the last segment's end. Text is
    compared after outer whitespace is stripped and the text is split into lines.
    """
    previous: _Cue | None = None
    for segment in segments:
        lines = segment.text.strip().split('\n')
        if previous is not None and lines == previous.lines:
            previous.end = segment.end
            continue
        if previous is not None:
            yield previous
        previous = _Cue(segment.start, segment.end, lines)
    if previous is not None:
        yield previous


def segments_to_srt(segments: Iterable[Segment]) -> str:
    """Serialize segments as the complete text of an SRT file.

    Cues are numbered from 1. Each cue is its number, its `start --> end` line and its text
    lines, followed by an empty line, so the text ends with a newline. No segments give an
    empty text.

    Args:
        segments: The segments in time order.

    Returns:
        The SRT text.

    Raises:
        ValueError: If a segment has a negative time.
    """
    lines: list[str] = []
    for number, cue in enumerate(_merged_cues(segments), start=1):
        lines.append(str(number))
        lines.append(f'{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}')
        lines.extend(cue.lines)
        lines.append('')
    return '\n'.join(lines)
