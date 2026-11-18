"""Check that a request's source language is a well-formed BCP 47 tag (DESIGN 5.1, 5.4).

The check is syntactic only. Whether the profile's engine supports the language is the adapter's
answer (`EngineAdapter.engine_language`). The grammar is the one that `youtube-collector`'s
`languages.py` uses at `9c1b957`.
"""

import re

__all__ = ['is_well_formed']

# RFC 5646 section 2.1, Language-Tag = langtag / privateuse / grandfathered. Matched
# case-insensitively; well-formedness only, so registry validity and duplicate subtags are not checked.
_LANGTAG = r"""
    (?:[a-z]{2,3}(?:-[a-z]{3}){0,3}|[a-z]{4}|[a-z]{5,8})   # language, with up to three extlangs
    (?:-[a-z]{4})?                                         # script
    (?:-(?:[a-z]{2}|[0-9]{3}))?                            # region
    (?:-(?:[a-z0-9]{5,8}|[0-9][a-z0-9]{3}))*               # variants
    (?:-[0-9a-wyz](?:-[a-z0-9]{2,8})+)*                    # extensions; the singleton is not x
    (?:-x(?:-[a-z0-9]{1,8})+)?                             # private use
"""
_PRIVATEUSE = r'x(?:-[a-z0-9]{1,8})+'
_GRANDFATHERED = (
    'en-gb-oed|i-ami|i-bnn|i-default|i-enochian|i-hak|i-klingon|i-lux|i-mingo|i-navajo|i-pwn|i-tao|i-tay|i-tsu'
    '|sgn-be-fr|sgn-be-nl|sgn-ch-de|art-lojban|cel-gaulish|no-bok|no-nyn|zh-guoyu|zh-hakka|zh-min|zh-min-nan|zh-xiang'
)
_LANGUAGE_TAG = re.compile(
    rf'(?:{_GRANDFATHERED})|(?:{_PRIVATEUSE})|(?:{_LANGTAG})', re.IGNORECASE | re.VERBOSE | re.ASCII,
)


def is_well_formed(tag: str) -> bool:
    """Return whether `tag` is a well-formed BCP 47 language tag.

    Any tag that the RFC 5646 grammar accepts is allowed, whatever its case. Registry validity and
    duplicate subtags are not checked.
    """
    return _LANGUAGE_TAG.fullmatch(tag) is not None
