"""Read media from, and publish SRT files into, the shared staging area (DESIGN 5.2, 5.6, 5.10).

The staging root is the SMB share that the collector writes into. Every path that leaves this
module is inside the root after symlinks are resolved. The transcriber only ever adds new SRT
files: publication goes through pathlib-extensions' `StagedFile` under `NEVER`, so an existing or
competing file is never replaced (M9). Where the filesystem has no hard links, such as the Mac's SMB
mount, a competing file that appears in the instant between the final check and the rename is
replaced instead (M21).

Nothing here decides a job's state. The worker (T7) maps these exceptions to failure categories.
"""

import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pathlib_extensions import OverwritePolicy, StagedFile

__all__ = [
    'MARKER_NAME',
    'InvalidPathError',
    'JobPaths',
    'MediaNotFoundError',
    'PathOutsideRootError',
    'StagingUnavailableError',
    'publish_srt',
    'require_marker',
    'resolve_job_paths',
    'srt_exists',
    'validate_request_path',
]

MARKER_NAME = '.media-transcriber-root'
"""The file that the user creates once in the staging root. The transcriber never creates it."""

WRITE_CHUNK_CHARS = 64 << 10
"""How many characters of SRT text to write between two calls of the checkpoint."""


class InvalidPathError(ValueError):
    """A request path is not a plain relative path inside the staging root (DESIGN 5.1)."""


class StagingUnavailableError(OSError):
    """The marker is missing or the root cannot be read, so the share is treated as unmounted."""


class MediaNotFoundError(FileNotFoundError):
    """The media file does not exist, or is not a file (failure category `missing`)."""


class PathOutsideRootError(ValueError):
    """A path resolves outside the staging root, for example through a symlink (DESIGN 5.2)."""


@dataclass(frozen=True)
class JobPaths:
    """The files of one job, both inside the staging root.

    Attributes:
        media: The requested media file, under the name that the request used.
        srt: The SRT file beside it: the requested path with the suffix `.srt`.
    """

    media: Path
    srt: Path


def _no_checkpoint() -> None:
    """Accept every step; the default when the caller needs no checks."""


def validate_request_path(path: str) -> PurePosixPath:
    """Check the syntax of a request path, without touching the filesystem (DESIGN 5.1).

    The API calls this, and it must not stall on a share that has stopped responding.

    Args:
        path: The path relative to the staging root, with `/` as the separator.

    Returns:
        The path, normalized by `PurePosixPath`.

    Raises:
        InvalidPathError: If the path is empty, absolute, contains a NUL character or a
            backslash, or has a component that is `..` or hidden (starts with a dot).
    """
    if '\0' in path or '\\' in path:
        raise InvalidPathError(f'A request path cannot contain a NUL character or a backslash: {path!r}')
    relative = PurePosixPath(path)
    if relative.is_absolute():
        raise InvalidPathError(f'A request path must be relative to the staging root: {path!r}')
    if not relative.parts:
        raise InvalidPathError('A request path cannot be empty')
    # A hidden component also covers `..`, and the hidden `.staging-*` directories of publication.
    if any(part.startswith('.') for part in relative.parts):
        raise InvalidPathError(f'A request path cannot contain `..` or a hidden component: {path!r}')
    return relative


def require_marker(root: Path) -> None:
    """Check that the staging root holds its marker file (DESIGN 5.10).

    On macOS an unmounted share leaves an empty directory or none under `/Volumes`, so only the
    marker shows that the share is mounted. The check reads one entry and never creates it.

    Args:
        root: The configured staging root.

    Raises:
        StagingUnavailableError: If the marker is missing or is not a file, or if the root cannot
            be read.
    """
    marker = root / MARKER_NAME
    try:
        mode = marker.stat().st_mode
    except OSError as error:
        raise StagingUnavailableError(f'The staging marker cannot be read: {marker}: {error}') from error
    if not stat.S_ISREG(mode):
        raise StagingUnavailableError(f'The staging marker is not a file: {marker}')


def resolve_job_paths(root: Path, relative: PurePosixPath) -> JobPaths:
    """Resolve a validated request path to the job's media and SRT paths inside the root.

    Both paths keep the requested names, so the SRT sits beside the name the user sees even when
    the media file is a symlink. Each path is checked after its symlinks are resolved, including a
    dangling symlink at the SRT path, whose target publication would create.

    Args:
        root: The configured staging root. Its marker has been checked.
        relative: A path from `validate_request_path`.

    Returns:
        The job's paths.

    Raises:
        MediaNotFoundError: If the media file does not exist or is not a file.
        PathOutsideRootError: If the media file or the SRT path resolves outside the root.
        OSError: If the root or the path cannot be resolved, for example in a symlink loop.
    """
    resolved_root = root.resolve(strict=True)
    media = root / relative
    try:
        resolved_media = media.resolve(strict=True)
    except FileNotFoundError:
        raise MediaNotFoundError(f'The media file does not exist: {relative}') from None
    if not resolved_media.is_relative_to(resolved_root):
        raise PathOutsideRootError(f'The media file resolves outside the staging root: {relative}')
    if not resolved_media.is_file():
        raise MediaNotFoundError(f'The media path is not a file: {relative}')
    srt = media.with_suffix('.srt')
    if not srt.resolve().is_relative_to(resolved_root):
        raise PathOutsideRootError(f'The SRT path resolves outside the staging root: {srt.relative_to(root)}')
    return JobPaths(media=media, srt=srt)


def srt_exists(paths: JobPaths) -> bool:
    """Apply the `NEVER` preflight: whether an SRT file already exists (DESIGN 5.5 step 3, M9).

    Only a file, or a symlink to one, at the exact SRT path counts. A dangling symlink does not.

    Args:
        paths: The job's paths.

    Returns:
        True if an SRT file exists, so the job ends `existing` before any decoding or model work.

    Raises:
        NotAFileError: If something other than a file, such as a directory, is at the SRT path.
        OSError: If the SRT path cannot be checked.
    """
    return OverwritePolicy.NEVER.resolve(paths.srt) is None


def publish_srt(paths: JobPaths, text: str, checkpoint: Callable[[], None] = _no_checkpoint) -> bool:
    """Publish complete SRT text atomically beside its media, never replacing a file (DESIGN 5.6).

    The text is written as UTF-8, without newline translation, into a fresh hidden `.staging-*`
    directory beside the SRT path, then hard-linked into place if nothing is there. Where the
    filesystem refuses hard links, it is renamed into place after a check that nothing is there,
    so a file appearing between that check and the rename is replaced (DESIGN M21). A failure or an
    exception leaves the SRT path untouched and removes the stage directory. An abrupt exit of the
    process can leave the stage directory behind, but never a partial SRT.

    Args:
        paths: The job's paths.
        text: The complete SRT text.
        checkpoint: Called before anything is written, between chunks of text, and once more
            immediately before the link. The worker passes the marker check (DESIGN 5.10, M18).
            An exception that it raises stops publication and propagates unchanged.

    Returns:
        True if the SRT was published. False if a file appeared at the SRT path during the job and
        was kept, so the job ends `existing`.

    Raises:
        MediaNotFoundError: If the media file is gone, so no SRT would sit beside it (M20).
        FileNotFoundError: If the directory that the SRT path resolves into does not exist.
            Publication never creates a directory.
        NotAFileError: If something other than a file is at the SRT path.
        FileExistsError: If something other than a file appeared at the SRT path meanwhile.
        OSError: If staging, writing or linking fails, for example on a filesystem without hard
            links.
    """
    checkpoint()
    if not paths.media.is_file():
        raise MediaNotFoundError(f'The media file disappeared before its SRT was published: {paths.media}')
    # StagedFile creates missing parents of the resolved target. A dangling SRT symlink can point
    # into a directory that does not exist, so the media check alone does not prevent that.
    if not paths.srt.resolve().parent.is_dir():
        raise FileNotFoundError(f'The directory of the SRT target does not exist: {paths.srt}')
    with StagedFile(paths.srt, OverwritePolicy.NEVER, rename_fallback=True) as staged:
        with staged.path.open('x', encoding='utf-8', newline='') as writer:
            for start in range(0, len(text), WRITE_CHUNK_CHARS):
                if start:
                    checkpoint()
                writer.write(text[start:start + WRITE_CHUNK_CHARS])
                writer.flush()
        # The last check comes after the text is complete and immediately before the link.
        checkpoint()
    return staged.published is not None
