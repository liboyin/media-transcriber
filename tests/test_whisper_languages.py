"""Tests for the Whisper language table and the mapping from BCP 47 tags."""

import pytest
from faster_whisper.tokenizer import _LANGUAGE_CODES

import media_transcriber.whisper_languages as testee


def test_whisper_languages_match_the_installed_faster_whisper() -> None:
    """The table lists exactly the codes that the installed faster-whisper accepts."""
    assert testee.WHISPER_LANGUAGES == frozenset(_LANGUAGE_CODES)


@pytest.mark.parametrize(('tag', 'code'), [
    ('en', 'en'),
    ('zh', 'zh'),
    ('zh-CN', 'zh'),
    ('zh-TW', 'zh'),
    ('ZH-Hant-TW', 'zh'),
    ('yue-HK', 'yue'),
    ('haw', 'haw'),
])
def test_whisper_language_maps_the_primary_subtag(tag: str, code: str) -> None:
    """The engine receives the lowercased primary language subtag (DESIGN 5.4)."""
    assert testee.whisper_language(tag) == code


@pytest.mark.parametrize('tag', ['jv', 'tlh', 'x-whatever', 'en_US'])
def test_whisper_language_returns_none_for_unsupported_languages(tag: str) -> None:
    """A primary subtag outside Whisper's codes is unsupported, with no aliases."""
    assert testee.whisper_language(tag) is None
