from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

class Segment:
    start: float
    end: float
    text: str

class TranscriptionInfo: ...

class WhisperModel:
    def __init__(
        self,
        model_size_or_path: str,
        device: str = ...,
        *,
        compute_type: str = ...,
        local_files_only: bool = ...,
    ) -> None: ...
    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        language: str | None = ...,
        task: str = ...,
    ) -> tuple[Iterable[Segment], TranscriptionInfo]: ...
