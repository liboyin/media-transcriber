"""Tests for SRT serialization against the source revision's formatting rules."""

import pytest

import media_transcriber.srt as testee
from media_transcriber.engine import Segment


@pytest.mark.parametrize(('seconds', 'expected'), [
    (0, '00:00:00,000'),
    (0.0004, '00:00:00,000'),
    (0.0006, '00:00:00,001'),
    (61.25, '00:01:01,250'),
    (59.9996, '00:01:00,000'),
    (3599.9996, '01:00:00,000'),
    (360_000, '100:00:00,000'),
])
def test_format_timestamp_rounds_milliseconds_before_decomposing(seconds: float, expected: str) -> None:
    """Rounding happens on the total, so a carry never yields 1000 milliseconds or 60 seconds."""
    assert testee.format_timestamp(seconds) == expected


def test_format_timestamp_rejects_a_negative_time() -> None:
    """A negative time raises instead of producing a malformed timestamp."""
    with pytest.raises(ValueError, match='negative'):
        testee.format_timestamp(-0.001)


def test_segments_to_srt_matches_the_source_revision_layout() -> None:
    """Cues are numbered, text is stripped and split into lines, and the text ends with one newline."""
    segments = [Segment(0.0, 1.5, '  Hello  '), Segment(1.5, 3.0, 'two\nlines\n')]

    assert testee.segments_to_srt(segments) == (
        '1\n00:00:00,000 --> 00:00:01,500\nHello\n\n'
        '2\n00:00:01,500 --> 00:00:03,000\ntwo\nlines\n'
    )


def test_segments_to_srt_merges_consecutive_identical_text() -> None:
    """A run of identical cues becomes one cue from the first start to the last end (M19)."""
    segments = [
        Segment(0.0, 1.0, 'again'),
        Segment(1.0, 2.0, ' again '),
        Segment(2.0, 3.0, 'again'),
        Segment(3.0, 4.0, 'other'),
        Segment(4.0, 5.0, 'again'),
    ]

    assert testee.segments_to_srt(segments) == (
        '1\n00:00:00,000 --> 00:00:03,000\nagain\n\n'
        '2\n00:00:03,000 --> 00:00:04,000\nother\n\n'
        '3\n00:00:04,000 --> 00:00:05,000\nagain\n'
    )


def test_segments_to_srt_of_no_segments_is_empty() -> None:
    """No segments serialize to an empty text, as at the source revision."""
    assert testee.segments_to_srt([]) == ''


def test_segments_to_srt_rejects_a_negative_time() -> None:
    """A segment with a negative time raises rather than producing a malformed cue."""
    with pytest.raises(ValueError, match='negative'):
        testee.segments_to_srt([Segment(-1.0, 1.0, 'x')])
