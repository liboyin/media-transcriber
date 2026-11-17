"""Tests for the whole-run rejection of known bad phrases."""

import numpy as np
import pytest
from fake_engine import FakeModel

import media_transcriber.rejection as testee
from media_transcriber.engine import Segment
from media_transcriber.rejection import RejectedTranscriptionError


def test_rejected_substrings_match_the_source_revision() -> None:
    """The phrases are exactly those of youtube-whisperer's rejection policy at 474f5dc."""
    assert testee.REJECTED_SUBSTRINGS == ('明镜与点点', '杨茜茜')


def test_reject_bad_segments_passes_clean_segments_through() -> None:
    """Segments without a rejected substring are re-yielded unchanged and in order."""
    segments = [Segment(0, 1, 'one'), Segment(1, 2, 'two')]

    assert list(testee.reject_bad_segments(segments)) == segments


@pytest.mark.parametrize('substring', testee.REJECTED_SUBSTRINGS)
def test_reject_bad_segments_raises_at_the_first_bad_segment_without_reading_ahead(substring: str) -> None:
    """Each phrase rejects the run as soon as its segment is reached, and later segments stay unread."""
    model = FakeModel([Segment(0, 1, 'fine'), Segment(1, 2, f'谢谢{substring}观看'), Segment(2, 3, 'unread')])
    rejected = testee.reject_bad_segments(model.transcribe(np.zeros(1, np.float32), 'zh'))

    assert next(rejected) == Segment(0, 1, 'fine')
    with pytest.raises(RejectedTranscriptionError) as raised:
        next(rejected)

    assert raised.value.substring == substring
    assert model.consumed == 2
