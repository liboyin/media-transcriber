"""The languages that Whisper models transcribe, and the mapping from BCP 47 tags (DESIGN 5.4).

Every Whisper engine shares this table: faster-whisper 1.2.1 and mlx-whisper 0.4.3 list the same
100 codes. A request states a BCP 47 tag, and the engine receives the tag's primary language
subtag, so `zh-CN`, `zh-TW` and `zh` all become `zh`.
"""

__all__ = ['WHISPER_LANGUAGES', 'whisper_language']

WHISPER_LANGUAGES = frozenset({
    'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs', 'ca', 'cs', 'cy', 'da',
    'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'gl', 'gu', 'ha', 'haw', 'he', 'hi',
    'hr', 'ht', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb',
    'ln', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'nn',
    'no', 'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sa', 'sd', 'si', 'sk', 'sl', 'sn', 'so', 'sq',
    'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 'vi',
    'yi', 'yo', 'yue', 'zh',
})
"""Whisper's own language codes. Most are ISO 639-1, but `haw` and `yue` are ISO 639-3, and
Javanese is `jw`, which matches no BCP 47 tag (`jv` is unsupported)."""


def whisper_language(tag: str) -> str | None:
    """Map a well-formed BCP 47 tag to the Whisper code of its primary language subtag.

    BCP 47 tags are case-insensitive, so the subtag is lowercased. The API (T8) checks that the
    tag is well-formed, so this function does not.

    Args:
        tag: The source language that the request states.

    Returns:
        The Whisper code, or None if Whisper does not support the language.
    """
    primary = tag.split('-', 1)[0].lower()
    return primary if primary in WHISPER_LANGUAGES else None
