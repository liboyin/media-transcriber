"""Tests for the BCP 47 well-formedness check of request languages (DESIGN 5.4)."""

import pytest

import media_transcriber.languages as testee


@pytest.mark.parametrize('tag', [
    'en', 'EN', 'zh', 'zh-CN', 'zh-cn', 'zh-Hans', 'zh-Hant-TW', 'es-419', 'sr-Latn-RS', 'fil', 'yue', 'abcd',
    'abcdefgh', 'zh-yue', 'zh-yue-abc-def', 'de-CH-1901', 'sl-rozaj-biske', 'de-1996', 'en-US-u-ca-gregory',
    'en-a-bbb-x-a-ccc', 'en-x-private', 'x-whatever', 'X-A', 'i-klingon', 'zh-min-nan', 'en-GB-oed', 'art-lojban',
    'de-DE-1901-1901',  # duplicate variants are well-formed, though not valid
])
def test_is_well_formed_accepts_rfc_5646_tags(tag: str) -> None:
    """Tags matching the RFC 5646 grammar are accepted, whatever their case."""
    assert testee.is_well_formed(tag)


@pytest.mark.parametrize('tag', [
    '', 'e', 'abcdefghi', 'en_US', 'en-', '-en', 'en--US', 'en US', ' en', 'en\n', 'zh-yue-abc-def-ghi', 'en-x',
    'en-a', 'en-a-b', 'en-x-abcdefghi', 'x', 'en-US-1', 'en-12', 'de-CH-190', 'en-ÜS', '中文', 'en-x-a-',
    'i-unknown',
])
def test_is_well_formed_rejects_malformed_tags(tag: str) -> None:
    """Strings outside the grammar, including empty subtags and non-ASCII, are rejected."""
    assert not testee.is_well_formed(tag)
