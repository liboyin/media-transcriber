"""The transcription pipeline that the worker and the benchmark both run (DESIGN 5.5, 7.3).

It covers DESIGN 5.5 steps 4 and 6 to 8 with the model already loaded: decoding, transcription,
rejection and SRT serialization. It never touches staging. The worker publishes the returned text,
and the benchmark writes it to a result set.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from media_transcriber.audio import SAMPLE_RATE, decode_waveform
from media_transcriber.engine import TranscriptionModel
from media_transcriber.rejection import reject_bad_segments
from media_transcriber.srt import segments_to_srt

__all__ = ['PipelineResult', 'run_pipeline']


@dataclass(frozen=True)
class PipelineResult:
    """The outcome of one successful pipeline run.

    Attributes:
        srt_text: The complete SRT text.
        segment_count: How many segments the engine produced, before cues were merged.
        audio_seconds: The duration of the decoded waveform.
        decoding_seconds: Time spent decoding the media file.
        transcription_seconds: Time spent producing and consuming every segment. The rejection
            check is included, because it runs while the lazy segments are consumed.
        serialization_seconds: Time spent turning the segments into SRT text.
    """

    srt_text: str
    segment_count: int
    audio_seconds: float
    decoding_seconds: float
    transcription_seconds: float
    serialization_seconds: float


def _no_checkpoint() -> None:
    """Accept every decoding chunk; the default when the caller needs no checks."""


def run_pipeline(
    media: Path,
    model: TranscriptionModel,
    language: str,
    checkpoint: Callable[[], None] = _no_checkpoint,
    clock: Callable[[], float] = time.perf_counter,
) -> PipelineResult:
    """Decode a media file, transcribe it, apply the rejection policy and serialize the result.

    Every segment is consumed, and the rejection check has passed, before serialization starts,
    so a rejected run produces no SRT text (DESIGN 5.5 steps 6 and 7).

    Args:
        media: The media file to decode.
        model: The loaded model.
        language: The engine's language code for the source language.
        checkpoint: Called between chunks of decoded audio, as `decode_waveform` describes.
        clock: A monotonic clock in seconds, for the stage timings.

    Returns:
        The SRT text, the counts and the timings.

    Raises:
        DecodingError: If ffmpeg fails.
        EmptyAudioError: If the media file decodes to no samples. No model work has started.
        RejectedTranscriptionError: If a segment contains a rejected substring.
        Exception: Whatever `checkpoint` or the model raises, unchanged.
    """
    started = clock()
    waveform = decode_waveform(media, checkpoint)
    decoded = clock()
    segments = list(reject_bad_segments(model.transcribe(waveform, language)))
    transcribed = clock()
    srt_text = segments_to_srt(segments)
    serialized = clock()
    return PipelineResult(
        srt_text=srt_text,
        segment_count=len(segments),
        audio_seconds=waveform.size / SAMPLE_RATE,
        decoding_seconds=decoded - started,
        transcription_seconds=transcribed - decoded,
        serialization_seconds=serialized - transcribed,
    )
