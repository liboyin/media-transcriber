"""Decode a media file's audio into the waveform that the engines transcribe (DESIGN 5.5 step 4).

ffmpeg runs as a child process and writes signed 16-bit PCM to a pipe, as `youtube-whisperer`'s
`waveform_loader.py` did at the source revision. The decoder reads the pipe in chunks so that the
worker can check the staging marker between them (DESIGN 5.11, M18).
"""

import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import IO

import numpy as np

from media_transcriber.engine import Waveform

__all__ = [
    'SAMPLE_RATE', 'DecodingError', 'DecodingStoppedError', 'EmptyAudioError', 'decode_waveform', 'kill_decoders',
]

SAMPLE_RATE = 16_000
"""Samples per second of every decoded waveform: Whisper's input rate."""

CHUNK_BYTES = 1 << 20
"""How many bytes of PCM to read between two calls of the checkpoint."""

STDERR_TAIL_BYTES = 8 << 10
"""How much of ffmpeg's standard error to keep for the error message."""

_running: set[subprocess.Popen[bytes]] = set()
"""The ffmpeg children that `decode_waveform` calls in this process currently own."""
_running_lock = threading.Lock()
_accepting = True
"""Whether `decode_waveform` may start ffmpeg. `kill_decoders` ends it for the rest of the process."""


class DecodingStoppedError(RuntimeError):
    """The process is shutting down, so no new decoder may start (DESIGN 5.9)."""


def kill_decoders() -> int:
    """Kill every running ffmpeg child, and refuse new ones, for a shutdown that abandons the worker (DESIGN 5.9).

    A decoder blocked on a stalled share would otherwise outlive the service and overlap with the
    next process. Starting ffmpeg and registering it happen under the same lock as this scan, so a
    decoder that a resumed worker tries to start afterwards never starts. The call that owns each
    killed child still reaps it and closes its pipes, if it ever resumes. Safe to call from any thread.

    Returns:
        How many children were killed.
    """
    global _accepting
    with _running_lock:
        _accepting = False
        processes = list(_running)
    for process in processes:
        process.kill()
    return len(processes)


class DecodingError(RuntimeError):
    """ffmpeg could not decode the media file.

    Attributes:
        returncode: ffmpeg's exit status.
        stderr: The end of ffmpeg's standard error, decoded leniently.
    """

    def __init__(self, path: Path, returncode: int, stderr: str) -> None:
        """Record the exit status and the diagnostics of a failed decode."""
        super().__init__(f'ffmpeg exited with status {returncode} while decoding {path}: {stderr.strip()}')
        self.returncode = returncode
        self.stderr = stderr


class EmptyAudioError(ValueError):
    """Decoding succeeded but produced no samples, so no model work may start (DESIGN 5.5 step 4)."""


def _no_checkpoint() -> None:
    """Accept every chunk; the default when the caller needs no checks between chunks."""


def _ffmpeg_command(path: Path) -> list[str]:
    """Build the ffmpeg arguments that decode `path` to mono 16 kHz signed 16-bit PCM on stdout.

    The `file:` prefix keeps ffmpeg from reading a file name such as `concat:x.mp4` as a protocol.
    `-nostdin` stops ffmpeg from reading the service's terminal, and `-loglevel error` keeps
    standard error short, so that its tail names the cause of a failure.
    """
    return [
        'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-threads', '0',
        '-i', f'file:{path.absolute()}',
        '-vn', '-f', 's16le', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', str(SAMPLE_RATE), 'pipe:1',
    ]


def _drain(stream: IO[bytes], tail: bytearray) -> None:
    """Read `stream` to its end, keeping its last `STDERR_TAIL_BYTES` bytes in `tail`.

    Draining stops ffmpeg from blocking on a full standard-error pipe while the decoder reads
    standard output.
    """
    for chunk in iter(lambda: stream.read(STDERR_TAIL_BYTES), b''):
        tail.extend(chunk)
        del tail[:-STDERR_TAIL_BYTES]


def decode_waveform(path: Path, checkpoint: Callable[[], None] = _no_checkpoint) -> Waveform:
    """Decode the audio of a media file into a mono 16 kHz float32 waveform.

    The child process is owned by this call: whether decoding succeeds, fails or is interrupted,
    ffmpeg has exited and been reaped, and its pipes are closed, before the call returns or raises.

    Args:
        path: The media file. ffmpeg decodes its first audio stream.
        checkpoint: Called after each chunk of decoded PCM. An exception that it raises stops
            ffmpeg and propagates unchanged.

    Returns:
        The samples, scaled from 16-bit integers to the range [-1, 1).

    Raises:
        DecodingError: If ffmpeg exits with a non-zero status, for example for a file without an
            audio stream.
        EmptyAudioError: If ffmpeg succeeds but produces no samples.
        OSError: If ffmpeg cannot be started.
        DecodingStoppedError: If `kill_decoders` has run in this process.
    """
    pcm = bytearray()
    stderr_tail = bytearray()
    with _running_lock:
        if not _accepting:
            raise DecodingStoppedError(f'The process is shutting down; not decoding {path}')
        process = subprocess.Popen(
            _ffmpeg_command(path), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        _running.add(process)
    stdout, stderr = process.stdout, process.stderr
    assert stdout is not None and stderr is not None  # both are pipes, as requested above
    drainer = threading.Thread(target=_drain, args=(stderr, stderr_tail), name='ffmpeg-stderr')
    try:
        drainer.start()
        for chunk in iter(lambda: stdout.read(CHUNK_BYTES), b''):
            pcm.extend(chunk)
            checkpoint()
        returncode = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        # The child has exited on every path here, which closes its end of standard error, so a
        # started drainer reaches the end of the stream and the join is bounded. A drainer that
        # failed to start has nothing to join. Closing our ends releases the descriptors.
        if drainer.ident is not None:
            drainer.join()
        stdout.close()
        stderr.close()
        with _running_lock:
            _running.discard(process)
    if returncode != 0:
        raise DecodingError(path, returncode, stderr_tail.decode(errors='replace'))
    # A whole number of 16-bit samples is expected; a stray odd byte cannot form a sample.
    samples = np.frombuffer(pcm, dtype=np.int16, count=len(pcm) // 2)
    if samples.size == 0:
        raise EmptyAudioError(f'No audio samples were decoded from {path}')
    return samples.astype(np.float32) / np.float32(32768)
