"""Engine-neutral types that every engine adapter implements (DESIGN section 8).

An adapter wraps one inference library. It maps a request's BCP 47 tag to the code its engine
understands, and loads a model for a profile. The loaded model turns a waveform into timed
segments. SRT is an export of those segments, never an engine's own output format (DESIGN 5.5).
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

__all__ = ['EngineAdapter', 'Profile', 'Segment', 'TranscriptionModel', 'Waveform']

Waveform = npt.NDArray[np.float32]
"""Mono 16 kHz samples in the range [-1, 1), as `media_transcriber.audio` decodes them."""


@dataclass(frozen=True)
class Segment:
    """One timed piece of transcribed text.

    Attributes:
        start: Start time in seconds from the beginning of the audio. Never negative.
        end: End time in seconds from the beginning of the audio. Never negative.
        text: The recognized text, as the engine produced it.
    """

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Profile:
    """A named, fixed combination of engine, model and decoding settings (DESIGN section 4).

    Profiles are the unit that a deployment offers and that the benchmark measures. A profile
    that uses turbo or distilled weights names them in `model`, so results never present them as
    `large-v3` (DESIGN section 8).

    Attributes:
        name: The name that requests and results use.
        engine: The name of the adapter that loads the profile.
        model: The model's identifier, such as a Hugging Face repository.
        revision: The model's pinned revision.
        precision: The numeric precision, such as `float32` or `float16`.
        decoding: Adapter-specific decoding settings. The adapter rejects keys it does not know.
    """

    name: str
    engine: str
    model: str
    revision: str
    precision: str
    decoding: Mapping[str, str | int | float | bool] = field(default_factory=dict)


class TranscriptionModel(Protocol):
    """A loaded model, ready to transcribe one waveform after another."""

    def transcribe(self, waveform: Waveform, language: str) -> Iterable[Segment]:
        """Transcribe a waveform in its stated source language.

        Args:
            waveform: The decoded audio.
            language: The engine's own language code, from `EngineAdapter.engine_language`.

        Returns:
            The segments in time order. They may be produced lazily, so the work is finished only
            when the caller has consumed every segment.
        """
        ...


class EngineAdapter(Protocol):
    """An adapter around one inference library."""

    name: str
    """The name that profiles use in `Profile.engine`."""

    def supported_languages(self) -> frozenset[str]:
        """Return the engine's language codes, which `GET /profiles` reports (DESIGN 5.3)."""
        ...

    def engine_language(self, tag: str) -> str | None:
        """Map a well-formed BCP 47 tag to the engine's language code.

        Args:
            tag: The source language that the request states (DESIGN 5.4).

        Returns:
            The engine's code, or None if the engine does not support the language.
        """
        ...

    def load(self, profile: Profile) -> TranscriptionModel:
        """Load the model that a profile names.

        Loading never downloads weights (DESIGN section 9).

        Args:
            profile: A profile whose `engine` is this adapter's name.

        Returns:
            The loaded model.
        """
        ...
