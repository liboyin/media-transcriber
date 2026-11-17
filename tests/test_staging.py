"""Tests for path resolution, the marker and atomic SRT publication in a test-owned staging root."""

import os
from pathlib import Path, PurePosixPath

import pytest
from pathlib_extensions import NotAFileError

import media_transcriber.staging as testee


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A staging root with its marker and one media file, `show/episode.mp4`."""
    staging = tmp_path / 'staging'
    (staging / 'show').mkdir(parents=True)
    (staging / testee.MARKER_NAME).touch()
    (staging / 'show' / 'episode.mp4').write_bytes(b'media')
    return staging


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    """A directory beside the staging root, holding one file, `secret.mp4`."""
    directory = tmp_path / 'outside'
    directory.mkdir()
    (directory / 'secret.mp4').write_bytes(b'secret')
    return directory


def job(root: Path, path: str = 'show/episode.mp4') -> testee.JobPaths:
    """Validate and resolve a request path, as the worker will."""
    return testee.resolve_job_paths(root, testee.validate_request_path(path))


def stage_dirs(directory: Path) -> list[Path]:
    """Return the publication stage directories left in `directory`."""
    return list(directory.glob('.staging-*'))


@pytest.mark.parametrize(('path', 'expected'), [
    ('episode.mp4', 'episode.mp4'),
    ('show/Title [id] - 1.mp4', 'show/Title [id] - 1.mp4'),
    ('show//episode.mp4', 'show/episode.mp4'),
    ('show/./episode.mp4', 'show/episode.mp4'),
])
def test_validate_request_path_accepts_plain_relative_paths(path: str, expected: str) -> None:
    """A plain relative path is accepted, normalized by PurePosixPath."""
    assert testee.validate_request_path(path) == PurePosixPath(expected)


@pytest.mark.parametrize('path', [
    '', '.', './', '/show/episode.mp4', '../episode.mp4', 'show/../../episode.mp4', 'show/..',
    '.hidden.mp4', 'show/.staging-abc/episode.mp4', testee.MARKER_NAME, 'show\\episode.mp4', 'show/epi\0sode.mp4',
])
def test_validate_request_path_rejects_unsafe_paths(path: str) -> None:
    """Empty, absolute, parent, hidden, backslash and NUL paths are rejected before any filesystem access."""
    with pytest.raises(testee.InvalidPathError):
        testee.validate_request_path(path)


def test_require_marker_accepts_a_root_with_its_marker(root: Path) -> None:
    """A root that holds the marker file is available."""
    testee.require_marker(root)


def test_require_marker_rejects_a_root_without_its_marker(root: Path) -> None:
    """A missing marker, as under an unmounted share's empty mount point, makes staging unavailable."""
    (root / testee.MARKER_NAME).unlink()

    with pytest.raises(testee.StagingUnavailableError, match='cannot be read'):
        testee.require_marker(root)


def test_require_marker_rejects_a_missing_root(tmp_path: Path) -> None:
    """A mount point that does not exist makes staging unavailable."""
    with pytest.raises(testee.StagingUnavailableError):
        testee.require_marker(tmp_path / 'absent')


def test_require_marker_rejects_a_marker_that_is_not_a_file(root: Path) -> None:
    """A directory with the marker's name does not count as the marker."""
    (root / testee.MARKER_NAME).unlink()
    (root / testee.MARKER_NAME).mkdir()

    with pytest.raises(testee.StagingUnavailableError, match='not a file'):
        testee.require_marker(root)


def test_require_marker_rejects_an_unreadable_root(root: Path) -> None:
    """A root that cannot be searched makes staging unavailable rather than raising PermissionError."""
    root.chmod(0)
    try:
        if os.access(root, os.X_OK):
            pytest.skip('the test user can search any directory, so permissions cannot be tested')
        with pytest.raises(testee.StagingUnavailableError):
            testee.require_marker(root)
    finally:
        root.chmod(0o755)


def test_require_marker_never_creates_the_marker(root: Path) -> None:
    """The check leaves a missing marker missing."""
    (root / testee.MARKER_NAME).unlink()

    with pytest.raises(testee.StagingUnavailableError):
        testee.require_marker(root)

    assert not (root / testee.MARKER_NAME).exists()


@pytest.mark.parametrize(('name', 'srt_name'), [
    ('episode.mp4', 'episode.srt'),
    ('Title v1.2.mkv', 'Title v1.2.srt'),
    ('noext', 'noext.srt'),
])
def test_resolve_job_paths_puts_the_srt_beside_the_media(root: Path, name: str, srt_name: str) -> None:
    """The SRT path is the requested path with its suffix replaced by .srt."""
    (root / 'show' / name).write_bytes(b'media')

    paths = job(root, f'show/{name}')

    assert paths == testee.JobPaths(media=root / 'show' / name, srt=root / 'show' / srt_name)


def test_resolve_job_paths_reports_missing_media(root: Path) -> None:
    """A media file that does not exist raises MediaNotFoundError, the `missing` category."""
    with pytest.raises(testee.MediaNotFoundError):
        job(root, 'show/absent.mp4')


def test_resolve_job_paths_reports_a_directory_as_missing_media(root: Path) -> None:
    """A directory at the media path is not a media file."""
    with pytest.raises(testee.MediaNotFoundError, match='not a file'):
        job(root, 'show')


def test_resolve_job_paths_reports_a_dangling_media_symlink_as_missing(root: Path) -> None:
    """A media symlink whose target is gone counts as missing media."""
    (root / 'show' / 'gone.mp4').symlink_to(root / 'show' / 'nowhere.mp4')

    with pytest.raises(testee.MediaNotFoundError):
        job(root, 'show/gone.mp4')


def test_resolve_job_paths_keeps_the_requested_name_of_a_symlink_inside_the_root(root: Path) -> None:
    """A media symlink to a file inside the root is accepted, and its SRT sits beside the link's name."""
    (root / 'show' / 'alias.mp4').symlink_to('episode.mp4')

    paths = job(root, 'show/alias.mp4')

    assert paths.srt == root / 'show' / 'alias.srt'


def test_resolve_job_paths_follows_a_root_that_is_a_symlink(tmp_path: Path, root: Path) -> None:
    """A staging root configured through a symlink, such as a mount alias, still contains its files."""
    alias = tmp_path / 'alias'
    alias.symlink_to(root)

    assert job(alias).media == alias / 'show' / 'episode.mp4'


@pytest.mark.parametrize(('link', 'target', 'requested'), [
    ('show/leak.mp4', '{outside}/secret.mp4', 'show/leak.mp4'),
    ('show/leak.mp4', '../../outside/secret.mp4', 'show/leak.mp4'),
    ('show/linkdir', '{outside}', 'show/linkdir/secret.mp4'),
], ids=['absolute-file', 'relative-file', 'directory'])
def test_resolve_job_paths_rejects_media_that_resolves_outside_the_root(
    root: Path, outside: Path, link: str, target: str, requested: str,
) -> None:
    """A symlink in the media path that leads outside the root is rejected (DESIGN 5.2)."""
    (root / link).symlink_to(target.format(outside=outside))

    with pytest.raises(testee.PathOutsideRootError, match='media file'):
        job(root, requested)


@pytest.mark.parametrize('target_name', ['secret.srt', 'secret.mp4'])
def test_resolve_job_paths_rejects_an_srt_symlink_that_leads_outside_the_root(
    root: Path, outside: Path, target_name: str,
) -> None:
    """An SRT symlink to a missing or an existing file outside the root is rejected before any work."""
    (root / 'show' / 'episode.srt').symlink_to(outside / target_name)

    with pytest.raises(testee.PathOutsideRootError, match='SRT path'):
        job(root)

    assert not (outside / 'secret.srt').exists()


def test_srt_exists_is_false_without_an_srt(root: Path) -> None:
    """With no file at the SRT path, the job goes on to transcription."""
    assert testee.srt_exists(job(root)) is False


def test_srt_exists_is_true_for_an_existing_srt(root: Path) -> None:
    """An existing SRT file ends the job `existing` (NEVER)."""
    (root / 'show' / 'episode.srt').write_text('old')

    assert testee.srt_exists(job(root)) is True


def test_srt_exists_counts_a_symlink_to_a_file_inside_the_root(root: Path) -> None:
    """A symlink at the SRT path to an existing file counts as an existing SRT."""
    (root / 'show' / 'other.srt').write_text('old')
    (root / 'show' / 'episode.srt').symlink_to('other.srt')

    assert testee.srt_exists(job(root)) is True


def test_srt_exists_raises_for_a_directory_at_the_srt_path(root: Path) -> None:
    """A directory at the SRT path cannot be published over, so it fails instead of looking absent."""
    (root / 'show' / 'episode.srt').mkdir()

    with pytest.raises(NotAFileError):
        testee.srt_exists(job(root))


def test_publish_srt_writes_the_exact_text_as_utf8(root: Path) -> None:
    """The published file holds the text's UTF-8 bytes, with no newline translation, and no stage is left."""
    text = '1\n00:00:00,000 --> 00:00:01,000\n你好 world\r\n'

    assert testee.publish_srt(job(root), text) is True

    assert (root / 'show' / 'episode.srt').read_bytes() == text.encode('utf-8')
    assert stage_dirs(root / 'show') == []


def test_publish_srt_publishes_an_empty_text(root: Path) -> None:
    """Audio with no speech publishes an empty SRT, as at the source revision."""
    assert testee.publish_srt(job(root), '') is True

    assert (root / 'show' / 'episode.srt').read_bytes() == b''


def test_publish_srt_keeps_an_existing_srt(root: Path) -> None:
    """An SRT that exists at publication is never replaced, and the call reports it as kept."""
    srt = root / 'show' / 'episode.srt'
    srt.write_text('old')

    assert testee.publish_srt(job(root), 'new') is False

    assert srt.read_text() == 'old'
    assert stage_dirs(root / 'show') == []


def test_publish_srt_keeps_a_file_that_appears_during_publication(root: Path) -> None:
    """A competing SRT created while the text is staged wins, and the call reports it as kept."""
    paths = job(root)
    srt = root / 'show' / 'episode.srt'
    calls: list[int] = []

    def competitor() -> None:
        calls.append(1)
        if len(calls) == 2:  # the last check, after the text is staged and before the link
            srt.write_text('competing')

    assert testee.publish_srt(paths, 'new', competitor) is False

    assert srt.read_text() == 'competing'
    assert stage_dirs(root / 'show') == []


def test_publish_srt_checks_before_writing_between_chunks_and_before_linking(
    root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The checkpoint runs first, between every two chunks, and once more after the text is complete."""
    monkeypatch.setattr(testee, 'WRITE_CHUNK_CHARS', 4)
    srt = root / 'show' / 'episode.srt'
    staged_lengths: list[int] = []

    def checkpoint() -> None:
        [stage] = stage_dirs(root / 'show') or [None]
        staged_lengths.append(len((stage / 'episode.srt').read_bytes()) if stage else -1)
        assert not srt.exists()

    assert testee.publish_srt(job(root), 'abcdefghij', checkpoint) is True

    # -1: before the stage exists. Then after 4 and 8 characters, and after all 10.
    assert staged_lengths == [-1, 4, 8, 10]


@pytest.mark.parametrize('failing_call', [1, 2, 4])
def test_publish_srt_publishes_nothing_when_the_checkpoint_raises(
    root: Path, monkeypatch: pytest.MonkeyPatch, failing_call: int,
) -> None:
    """A failed check before, during or after the write publishes nothing and removes the stage."""
    monkeypatch.setattr(testee, 'WRITE_CHUNK_CHARS', 4)
    calls: list[int] = []

    class Unavailable(Exception):
        """The checkpoint's deliberate failure."""

    def checkpoint() -> None:
        calls.append(1)
        if len(calls) == failing_call:
            raise Unavailable

    with pytest.raises(Unavailable):
        testee.publish_srt(job(root), 'abcdefghij', checkpoint)

    assert not (root / 'show' / 'episode.srt').exists()
    assert stage_dirs(root / 'show') == []


def test_publish_srt_stops_when_the_marker_disappears(root: Path) -> None:
    """With the marker check as the checkpoint, an unmounted share stops publication before any write."""
    paths = job(root)
    (root / testee.MARKER_NAME).unlink()

    with pytest.raises(testee.StagingUnavailableError):
        testee.publish_srt(paths, 'text', lambda: testee.require_marker(root))

    assert not (root / 'show' / 'episode.srt').exists()
    assert stage_dirs(root / 'show') == []


def test_publish_srt_refuses_when_the_media_file_is_gone(root: Path) -> None:
    """Media moved away during the job fails as missing, and no SRT is left without it (M20)."""
    paths = job(root)
    (root / 'show' / 'episode.mp4').unlink()

    with pytest.raises(testee.MediaNotFoundError, match='disappeared'):
        testee.publish_srt(paths, 'text')

    assert not (root / 'show' / 'episode.srt').exists()
    assert stage_dirs(root / 'show') == []


def test_publish_srt_never_recreates_a_removed_directory(root: Path) -> None:
    """If the media's whole directory was moved away, publication creates no directory in its place."""
    paths = job(root)
    (root / 'show' / 'episode.mp4').unlink()
    (root / 'show').rmdir()

    with pytest.raises(testee.MediaNotFoundError):
        testee.publish_srt(paths, 'text')

    assert not (root / 'show').exists()
