from typing import ClassVar, TypedDict

import numpy as np
import numpy.typing as npt
from mlx.core import Dtype
from mlx_whisper.whisper import Whisper

class ModelHolder:
    model: ClassVar[Whisper | None]
    model_path: ClassVar[str | None]
    @classmethod
    def get_model(cls, model_path: str, dtype: Dtype) -> Whisper: ...

# Stub-only: mlx-whisper returns plain dicts with more keys than these.
class _Segment(TypedDict):
    start: float
    end: float
    text: str

class _Result(TypedDict):
    segments: list[_Segment]

def transcribe(
    audio: npt.NDArray[np.float32],
    *,
    path_or_hf_repo: str = ...,
    language: str | None = ...,
    task: str = ...,
    fp16: bool = ...,
) -> _Result: ...
