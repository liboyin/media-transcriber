"""Tests for the worker, with real staging and ffmpeg on synthetic audio, and fake engines (DESIGN 5.5, 5.9-5.11)."""

import gc
import logging
import sqlite3
import threading
import wave
import weakref
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from conftest import ThreadOwner
from fake_engine import FakeAdapter, FakeModel

import media_transcriber.worker as testee
from media_transcriber.audio import DecodingError, EmptyAudioError
from media_transcriber.engine import Profile, Segment
from media_transcriber.rejection import REJECTED_SUBSTRINGS, RejectedTranscriptionError
from media_transcriber.staging import MARKER_NAME, MediaNotFoundError, PathOutsideRootError, StagingUnavailableError
from media_transcriber.store import Category, Phase, QueuedJob, Store, initialize

PROFILE = Profile(name='p', engine='fake', model='org/model', revision='r1', precision='float32')
OTHER_PROFILE = Profile(name='q', engine='fake', model='org/other', revision='r2', precision='float32')
SEGMENTS = [Segment(0.0, 0.1, 'beep')]
SRT_TEXT = '1\n00:00:00,000 --> 00:00:00,100\nbeep\n'
WAIT_SECONDS = 10.0


class Ticks:
    """A monotonic clock that advances one second on every reading, so every span is measurable."""

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Advance and return the time."""
        self.now += 1.0
        return self.now


class Service:
    """A worker with test-owned staging, database, adapter and clock."""

    def __init__(self, tmp_path: Path) -> None:
        """Create the staging root with its marker, the database and a worker that records fatal errors."""
        self.root = tmp_path / 'staging'
        self.root.mkdir()
        self.marker = self.root / MARKER_NAME
        self.marker.touch()
        self.db_path = tmp_path / 'db.sqlite3'
        initialize(self.db_path)
        self.store = Store(self.db_path, lambda: 1_000_000.0)
        self.model = FakeModel(SEGMENTS)
        self.adapter = FakeAdapter(model=self.model)
        self.clock = Ticks()
        self.fatal: list[BaseException] = []
        self.worker = self.make_worker()

    def make_worker(self, **options: object) -> testee.Worker:
        """Build a worker on this service's collaborators, with optional overrides."""
        arguments: dict[str, object] = {
            'open_store': lambda: Store(self.db_path, lambda: 1_000_000.0), 'staging_root': self.root,
            'profiles': {PROFILE.name: PROFILE, OTHER_PROFILE.name: OTHER_PROFILE},
            'adapters': {'fake': self.adapter}, 'monotonic': self.clock, 'on_fatal': self.fatal.append,
            **options,
        }
        return testee.Worker(**arguments)  # type: ignore[arg-type]

    def media(self, name: str = 'video.wav', samples: int = 1600) -> str:
        """Write a mono 16 kHz WAV clip into staging and return its relative path."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), 'wb') as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(np.full(samples, 1000, dtype='<i2').tobytes())
        return name

    def submit(self, path: str, profile: str = PROFILE.name) -> str:
        """Queue a job and return its ID."""
        job = self.store.submit(path=path, language='en-GB', engine_language='en', profile=profile,
                                retention_seconds=100.0)
        return str(job['id'])

    def job(self, job_id: str) -> dict[str, object]:
        """Return a job's representation."""
        job = self.store.get_job(job_id)
        assert job is not None
        return job

    def failure(self, job_id: str) -> tuple[object, object]:
        """Return the phase and category of a failed job's failure."""
        job = self.job(job_id)
        assert job['state'] == 'failed', job
        failure = job['failure']
        assert isinstance(failure, dict)
        return failure['phase'], failure['category']


@pytest.fixture
def service(tmp_path: Path) -> Iterator[Service]:
    """Provide a worker on test-owned storage; the test's store is closed afterwards."""
    owned = Service(tmp_path)
    try:
        yield owned
    finally:
        owned.store.close()


def stage_entries(root: Path) -> list[str]:
    """Return the names of hidden stage directories that publication left behind."""
    return sorted(path.name for path in root.rglob('.staging-*'))


# staging_stale


@pytest.mark.parametrize(('name', 'checked_at', 'stale'), [
    (testee.ActivityName.IDLE, 80.0, False),
    (testee.ActivityName.IDLE, 79.0, True),
    (testee.ActivityName.DECODING, 79.0, True),
    (testee.ActivityName.PUBLISHING, 79.0, True),
    (testee.ActivityName.STAGING, 79.0, True),
    (testee.ActivityName.PAUSED, 79.0, True),
    (testee.ActivityName.TRANSCRIBING, 0.0, False),
    (testee.ActivityName.LOADING, 0.0, False),
])
def test_staging_stale_applies_the_limit_only_to_share_activities(
    name: testee.ActivityName, checked_at: float, stale: bool,
) -> None:
    """A check older than the limit is stale while the share is in use, never during in-memory work (M17)."""
    activity = testee.Activity(name, None, 0.0, 0.0)
    assert testee.staging_stale(testee.StagingCheck(True, checked_at), activity, now=100.0, limit=20.0) is stale


def test_staging_stale_treats_no_check_as_stale_during_share_activities() -> None:
    """Before the first check, a share activity has nothing to vouch for the share."""
    idle = testee.Activity(testee.ActivityName.IDLE, None, 0.0, 0.0)
    loading = testee.Activity(testee.ActivityName.LOADING, None, 0.0, 0.0)
    assert testee.staging_stale(None, idle, now=1.0, limit=20.0)
    assert not testee.staging_stale(None, loading, now=1.0, limit=20.0)


# failure_for


@pytest.mark.parametrize(('phase', 'error', 'expected'), [
    (Phase.STAGING, MediaNotFoundError('gone'), (Phase.STAGING, Category.MISSING)),
    (Phase.PUBLICATION, MediaNotFoundError('gone'), (Phase.PUBLICATION, Category.MISSING)),
    (Phase.DECODING, StagingUnavailableError('no marker'), (Phase.DECODING, Category.FILESYSTEM)),
    (Phase.PUBLICATION, StagingUnavailableError('no marker'), (Phase.PUBLICATION, Category.FILESYSTEM)),
    (Phase.TRANSCRIPTION, RejectedTranscriptionError('x'), (Phase.REJECTION, Category.REJECTED)),
    (Phase.MODEL, ValueError('bad precision'), (Phase.MODEL, Category.MODEL)),
    (Phase.MODEL, FileNotFoundError('weights'), (Phase.MODEL, Category.MODEL)),
    (Phase.DECODING, DecodingError(Path('m'), 1, 'bad'), (Phase.DECODING, Category.DECODING)),
    (Phase.DECODING, EmptyAudioError('empty'), (Phase.DECODING, Category.DECODING)),
    # Starting a missing ffmpeg binary raises a plain FileNotFoundError: not the media's absence.
    (Phase.DECODING, FileNotFoundError('ffmpeg'), (Phase.DECODING, Category.INTERNAL)),
    (Phase.STAGING, PathOutsideRootError('escape'), (Phase.STAGING, Category.FILESYSTEM)),
    (Phase.STAGING, PermissionError('denied'), (Phase.STAGING, Category.FILESYSTEM)),
    (Phase.PUBLICATION, FileNotFoundError('no directory'), (Phase.PUBLICATION, Category.FILESYSTEM)),
    (Phase.PUBLICATION, FileExistsError('directory appeared'), (Phase.PUBLICATION, Category.FILESYSTEM)),
    (Phase.TRANSCRIPTION, RuntimeError('device lost'), (Phase.TRANSCRIPTION, Category.INTERNAL)),
    (Phase.TRANSCRIPTION, OSError('engine I/O'), (Phase.TRANSCRIPTION, Category.INTERNAL)),
    (Phase.STAGING, ValueError('bad suffix'), (Phase.STAGING, Category.INTERNAL)),
])
def test_failure_for_maps_each_error_to_its_phase_and_category(
    phase: Phase, error: Exception, expected: tuple[Phase, Category],
) -> None:
    """Each error maps to the category of DESIGN 5.8, with the specific staging errors first."""
    failure = testee.failure_for(phase, error)
    assert (failure.phase, failure.category) == expected


def test_failure_for_keeps_the_type_and_raw_message() -> None:
    """The message names the exception type and keeps its text, or the type alone when it has none."""
    assert testee.failure_for(Phase.MODEL, ValueError('bad')).message == 'ValueError: bad'
    assert testee.failure_for(Phase.TRANSCRIPTION, MemoryError()).message == 'MemoryError'


# A job's steps


def test_step_transcribes_and_publishes_a_job(service: Service) -> None:
    """A queued job runs to `done` with `transcribed`, and its complete SRT sits beside the media (DESIGN 5.5)."""
    job_id = service.submit(service.media('sub/video.wav'))

    assert service.worker.step(service.store) is testee.Step.PROCESSED

    assert (service.root / 'sub/video.srt').read_text() == SRT_TEXT
    job = service.job(job_id)
    assert (job['state'], job['outcome'], job['srt_path'], job['phase']) == ('done', 'transcribed',
                                                                             'sub/video.srt', 'publication')
    durations = job['durations']
    assert isinstance(durations, dict)
    assert all(isinstance(durations[name], float) and durations[name] > 0
               for name in ('model_load', 'decoding', 'transcription', 'publication'))
    [(_, language)] = service.model.calls
    assert language == 'en'
    assert stage_entries(service.root) == []
    assert service.worker.activity.name is testee.ActivityName.IDLE


def test_step_is_idle_when_nothing_is_queued(service: Service) -> None:
    """Without a job the worker idles, after a marker check that keeps health fresh."""
    assert service.worker.step(service.store) is testee.Step.IDLE
    check = service.worker.staging_check
    assert check is not None and check.marker_present
    assert service.worker.activity == testee.Activity(testee.ActivityName.IDLE, None, 1.0, 1.0)


def test_an_existing_srt_ends_the_job_before_any_decoding_or_model_work(service: Service) -> None:
    """The `NEVER` preflight finds the SRT, so nothing is decoded, loaded or written (DESIGN 5.5 step 3, M9)."""
    job_id = service.submit(service.media())
    (service.root / 'video.srt').write_text('kept')

    with patch.object(testee, 'run_pipeline', side_effect=AssertionError('decoded')):
        service.worker.step(service.store)

    job = service.job(job_id)
    assert (job['state'], job['outcome'], job['srt_path']) == ('done', 'existing', 'video.srt')
    assert (service.root / 'video.srt').read_text() == 'kept'
    assert service.adapter.loads == []


def test_a_competing_srt_that_appears_during_the_job_is_kept(service: Service) -> None:
    """A file that appears at the SRT path before publication is preserved, and the job ends `existing`."""
    job_id = service.submit(service.media())
    service.model.on_transcribe.append(lambda: (service.root / 'video.srt').write_text('competitor'))

    service.worker.step(service.store)

    job = service.job(job_id)
    assert (job['state'], job['outcome']) == ('done', 'existing')
    assert (service.root / 'video.srt').read_text() == 'competitor'


def test_the_model_stays_resident_across_jobs_of_one_profile(service: Service) -> None:
    """Two jobs of one profile load its model once (DESIGN 5.3)."""
    service.submit(service.media('one.wav'))
    service.submit(service.media('two.wav'))
    service.worker.step(service.store)
    service.worker.step(service.store)
    assert service.adapter.loads == ['p']
    assert service.worker.resident_profile == 'p'
    assert len(service.model.calls) == 2


def test_another_profile_replaces_the_resident_model_after_dropping_it(service: Service) -> None:
    """A job for another profile loads it, and no model is resident while it loads (DESIGN 5.3)."""
    service.submit(service.media('one.wav'))
    service.submit(service.media('two.wav'), profile='q')
    resident_during_load: list[str | None] = []
    service.adapter.on_load.append(lambda: resident_during_load.append(service.worker.resident_profile))

    service.worker.step(service.store)
    service.worker.step(service.store)

    assert service.adapter.loads == ['p', 'q']
    assert resident_during_load == [None, None]
    assert service.worker.resident_profile == 'q'


def test_a_failed_load_fails_the_job_model_and_leaves_no_model_resident(service: Service) -> None:
    """A load error fails the job in the phase `model`; the next job loads again."""
    service.submit(service.media('one.wav'))
    service.worker.step(service.store)
    failing = service.submit(service.media('two.wav'), profile='q')
    service.adapter.error = FileNotFoundError('no weights')

    service.worker.step(service.store)

    assert service.failure(failing) == ('model', 'model')
    assert service.worker.resident_profile is None
    assert not (service.root / 'two.srt').exists()


@pytest.mark.parametrize(('profiles', 'adapters', 'message'), [
    pytest.param({}, None, "does not configure the profile 'p'", id='profile'),
    pytest.param(None, {}, "no adapter for the engine 'fake'", id='engine'),
])
def test_a_job_for_an_unconfigured_profile_or_engine_fails_model(
    service: Service, profiles: dict[str, Profile] | None, adapters: dict[str, FakeAdapter] | None, message: str,
) -> None:
    """A queued job whose profile or engine the deployment no longer offers fails `model` after decoding."""
    options: dict[str, object] = {}
    if profiles is not None:
        options['profiles'] = profiles
    if adapters is not None:
        options['adapters'] = adapters
    worker = service.make_worker(**options)
    job_id = service.submit(service.media())

    worker.step(service.store)

    assert service.failure(job_id) == ('model', 'model')
    failure = service.job(job_id)['failure']
    assert isinstance(failure, dict) and message in str(failure['message'])


def test_a_missing_media_file_fails_the_job_missing(service: Service) -> None:
    """A path with no file behind it fails in the phase `staging` with category `missing`."""
    job_id = service.submit('absent.wav')
    service.worker.step(service.store)
    assert service.failure(job_id) == ('staging', 'missing')


def test_media_that_disappears_before_publication_fails_missing_and_writes_nothing(service: Service) -> None:
    """Media moved away during the job leaves no SRT behind, and the job fails `missing` (M20)."""
    job_id = service.submit(service.media())
    service.model.on_transcribe.append((service.root / 'video.wav').unlink)

    service.worker.step(service.store)

    assert service.failure(job_id) == ('publication', 'missing')
    assert not (service.root / 'video.srt').exists()
    assert stage_entries(service.root) == []


def test_a_symlink_that_escapes_the_root_fails_filesystem(service: Service, tmp_path: Path) -> None:
    """Media that resolves outside the staging root is never read (DESIGN 5.2)."""
    outside = tmp_path / 'outside.wav'
    outside.write_bytes(b'x')
    (service.root / 'link.wav').symlink_to(outside)
    job_id = service.submit('link.wav')
    service.worker.step(service.store)
    assert service.failure(job_id) == ('staging', 'filesystem')
    assert service.adapter.loads == []


def test_a_directory_at_the_srt_path_fails_filesystem(service: Service) -> None:
    """Something other than a file at the SRT path fails the job before decoding."""
    job_id = service.submit(service.media())
    (service.root / 'video.srt').mkdir()
    service.worker.step(service.store)
    assert service.failure(job_id) == ('staging', 'filesystem')


def test_undecodable_media_fails_decoding_without_model_work(service: Service) -> None:
    """A file ffmpeg cannot decode fails in the phase `decoding`, before any load."""
    (service.root / 'bad.wav').write_bytes(b'not audio')
    job_id = service.submit('bad.wav')
    service.worker.step(service.store)
    assert service.failure(job_id) == ('decoding', 'decoding')
    assert service.adapter.loads == []


def test_empty_audio_fails_decoding_without_model_work(service: Service) -> None:
    """A clip with no samples fails `decoding` before the model is loaded (DESIGN 5.5 step 4)."""
    job_id = service.submit(service.media(samples=0))
    service.worker.step(service.store)
    assert service.failure(job_id) == ('decoding', 'decoding')
    assert service.adapter.loads == []


def test_a_rejected_transcription_fails_rejected_and_writes_nothing(service: Service) -> None:
    """A rejected phrase fails the whole run in the phase `rejection`, and no SRT exists (DESIGN 5.5 step 7)."""
    service.model.segments = [Segment(0.0, 0.1, REJECTED_SUBSTRINGS[0])]
    job_id = service.submit(service.media())
    service.worker.step(service.store)
    assert service.failure(job_id) == ('rejection', 'rejected')
    assert not (service.root / 'video.srt').exists()


def test_an_engine_error_during_transcription_fails_internal(service: Service) -> None:
    """An error from the engine while it transcribes fails `internal` in the phase `transcription`."""
    service.model.error = RuntimeError('device lost')
    job_id = service.submit(service.media())
    service.worker.step(service.store)
    assert service.failure(job_id) == ('transcription', 'internal')
    durations = service.job(job_id)['durations']
    assert isinstance(durations, dict) and durations['publication'] is None and durations['decoding'] > 0


# The staging marker


def test_a_missing_marker_pauses_without_changing_any_record(
    service: Service, caplog: pytest.LogCaptureFixture,
) -> None:
    """While the marker is missing nothing starts or fails, and work resumes once it returns (DESIGN 5.10)."""
    job_id = service.submit(service.media())
    before = service.job(job_id)
    service.marker.unlink()

    with caplog.at_level(logging.WARNING, logger=testee.__name__):
        assert service.worker.step(service.store) is testee.Step.PAUSED
        assert service.worker.step(service.store) is testee.Step.PAUSED

    assert service.job(job_id) == before
    check = service.worker.staging_check
    assert check is not None and not check.marker_present
    assert service.worker.activity.name is testee.ActivityName.PAUSED
    assert caplog.text.count('pausing work') == 1

    service.marker.touch()
    with caplog.at_level(logging.WARNING, logger=testee.__name__):
        assert service.worker.step(service.store) is testee.Step.PROCESSED
    assert 'resuming work' in caplog.text
    assert service.job(job_id)['state'] == 'done'


def test_a_marker_that_disappears_during_decoding_fails_filesystem(service: Service) -> None:
    """Decoding checks the marker again between chunks, and its absence fails the job (DESIGN 5.10, M18)."""
    worker = service.make_worker(marker_check_interval=0.0)
    job_id = service.submit(service.media())
    real_resolve = testee.resolve_job_paths

    def resolve_then_unmount(root: Path, relative: object) -> object:
        paths = real_resolve(root, relative)  # type: ignore[arg-type]
        service.marker.unlink()
        return paths

    with patch.object(testee, 'resolve_job_paths', resolve_then_unmount):
        worker.step(service.store)

    assert service.failure(job_id) == ('decoding', 'filesystem')
    assert service.adapter.loads == []


def test_decoding_checks_the_marker_only_once_the_interval_has_passed(service: Service) -> None:
    """Between chunks, the marker is checked again only when the last check is older than the interval."""
    checks: list[Path] = []
    real_require = testee.require_marker

    def counting(root: Path) -> None:
        checks.append(root)
        real_require(root)

    service.submit(service.media(samples=16000 * 70))  # more than two chunks of PCM
    with patch.object(testee, 'require_marker', counting):
        service.make_worker(marker_check_interval=1e9).step(service.store)
        relaxed = len(checks)
        checks.clear()
        service.submit(service.media('two.wav', samples=16000 * 70))
        service.make_worker(marker_check_interval=0.0).step(service.store)

    # The step's own check, the check before publication resolves the paths again, then publication's
    # checks: before writing and before linking.
    assert relaxed == 4
    assert len(checks) == relaxed + 3  # plus one check after each of the three decoded chunks


def test_a_marker_that_disappears_before_publication_publishes_nothing(service: Service) -> None:
    """Publication checks the marker first, so an unmounted share gets no SRT and the job fails (DESIGN 5.10)."""
    job_id = service.submit(service.media())
    service.model.on_transcribe.append(service.marker.unlink)

    service.worker.step(service.store)

    assert service.failure(job_id) == ('publication', 'filesystem')
    assert not (service.root / 'video.srt').exists()
    assert stage_entries(service.root) == []
    check = service.worker.staging_check
    assert check is not None and not check.marker_present


def test_a_marker_that_disappears_during_publication_publishes_nothing(service: Service) -> None:
    """Publication's own checkpoints check the marker, after the checks that precede it (DESIGN 5.10)."""
    job_id = service.submit(service.media())
    real_resolve = testee.resolve_job_paths
    calls: list[object] = []

    def resolve_then_unmount_on_the_second_call(root: Path, relative: object) -> object:
        paths = real_resolve(root, relative)  # type: ignore[arg-type]
        calls.append(relative)
        if len(calls) == 2:  # the resolution that immediately precedes publication
            service.marker.unlink()
        return paths

    with patch.object(testee, 'resolve_job_paths', resolve_then_unmount_on_the_second_call):
        service.worker.step(service.store)

    assert len(calls) == 2
    assert service.failure(job_id) == ('publication', 'filesystem')
    assert not (service.root / 'video.srt').exists()
    assert stage_entries(service.root) == []


# Activity and progress


def test_the_activity_follows_the_job_and_reports_progress(service: Service) -> None:
    """Each phase publishes its activity for the job; decoding and publication report progress (DESIGN 5.11)."""
    seen: list[testee.Activity] = []
    service.adapter.on_load.append(lambda: seen.append(service.worker.activity))
    service.model.on_transcribe.append(lambda: seen.append(service.worker.activity))
    real_publish = testee.publish_srt

    def spy_publish(paths: object, text: str, checkpoint: Callable[[], None]) -> bool:
        seen.append(service.worker.activity)
        checkpoint()
        seen.append(service.worker.activity)
        return real_publish(paths, text, checkpoint)  # type: ignore[arg-type]

    job_id = service.submit(service.media())
    with patch.object(testee, 'publish_srt', spy_publish):
        service.worker.step(service.store)

    loading, transcribing, publishing, progressed = seen
    assert [activity.name for activity in seen] == ['loading', 'transcribing', 'publishing', 'publishing']
    assert all(activity.job_id == job_id for activity in seen)
    assert [activity.touches_share for activity in seen] == [False, False, True, True]
    assert loading.started_at < transcribing.started_at < publishing.started_at
    assert progressed.started_at == publishing.started_at < progressed.progress_at
    assert service.worker.activity.name is testee.ActivityName.IDLE
    assert service.worker.activity.job_id is None


def test_decoding_reports_progress_without_refreshing_the_marker_check(service: Service) -> None:
    """A decoded chunk updates progress, but within the interval it does not touch the marker check (M18)."""
    worker = service.make_worker(marker_check_interval=1e9)
    seen: list[tuple[testee.Activity, testee.StagingCheck | None]] = []

    def fake_pipeline(media: Path, load: Callable[[], object], language: str, checkpoint: Callable[[], None],
                      clock: Callable[[], float]) -> object:
        for _ in range(2):
            checkpoint()
            seen.append((worker.activity, worker.staging_check))
        raise RuntimeError('stop after decoding')

    service.submit(service.media())
    with patch.object(testee, 'run_pipeline', fake_pipeline):
        worker.step(service.store)

    (first, first_check), (second, second_check) = seen
    assert first.name is second.name is testee.ActivityName.DECODING
    assert first.started_at == second.started_at < first.progress_at < second.progress_at
    assert first_check is second_check  # the step's own check, not refreshed by progress


# Store faults and the run loop


class FailingPhaseStore(Store):
    """A store whose `set_phase` fails for one phase, as a broken database would."""

    fail_phase: Phase = Phase.DECODING

    def set_phase(self, job_id: str, phase: Phase) -> None:
        """Fail for the chosen phase, and record the others."""
        if phase is self.fail_phase:
            raise sqlite3.OperationalError('disk I/O error')
        super().set_phase(job_id, phase)


@pytest.mark.parametrize('phase', [Phase.DECODING, Phase.MODEL, Phase.TRANSCRIPTION])
def test_a_store_error_during_a_job_is_fatal_and_never_a_job_failure(
    service: Service, owned_threads: ThreadOwner, phase: Phase,
) -> None:
    """A failed phase write ends the worker with the store's error; the job stays running for recovery."""

    def open_store() -> Store:
        # Opened on the worker thread, which alone may use the connection.
        store = FailingPhaseStore(service.db_path, lambda: 1_000_000.0)
        store.fail_phase = phase
        return store

    ended = threading.Event()
    fatal: list[BaseException] = []

    def on_fatal(error: BaseException) -> None:
        fatal.append(error)
        ended.set()

    worker = service.make_worker(open_store=open_store, on_fatal=on_fatal)
    job_id = service.submit(service.media())

    thread = owned_threads.start(worker.run, stop=worker.stop)
    assert ended.wait(WAIT_SECONDS)
    thread.join(WAIT_SECONDS)

    [error] = fatal
    assert isinstance(error, sqlite3.OperationalError)
    assert worker.failed
    job = service.job(job_id)
    assert (job['state'], job['failure']) == ('running', None)
    assert service.store.list_failures() == []


def test_run_recovers_processes_and_stops(service: Service, owned_threads: ThreadOwner) -> None:
    """The loop requeues an interrupted job, runs it, then idles until stopped (DESIGN 5.9)."""
    job_id = service.submit(service.media())
    service.store.start_job(job_id)
    worker = service.make_worker(idle_wait=WAIT_SECONDS)
    finished = threading.Event()
    service.model.on_transcribe.append(finished.set)

    thread = owned_threads.start(worker.run, stop=worker.stop)
    assert finished.wait(WAIT_SECONDS)
    worker.stop()
    thread.join(WAIT_SECONDS)

    assert not thread.is_alive()
    assert service.job(job_id)['state'] == 'done'
    assert service.fatal == [] and not worker.failed


class SignallingStore(Store):
    """A store that signals each time the worker finds the queue empty."""

    empty: threading.Event

    def next_job(self) -> QueuedJob | None:
        """Return the next job, and signal when there is none."""
        job = super().next_job()
        if job is None:
            self.empty.set()
        return job


def wait_until(condition: Callable[[], bool], description: str) -> None:
    """Poll a condition that another thread brings about, failing after a bound."""
    pause = threading.Event()
    for _ in range(int(WAIT_SECONDS / 0.01)):
        if condition():
            return
        pause.wait(0.01)
    raise AssertionError(f'timed out waiting for {description}')


def test_wake_ends_an_idle_wait_early(service: Service, owned_threads: ThreadOwner) -> None:
    """A submission's wake-up starts the job at once, not after the idle wait."""
    empty = threading.Event()

    def open_store() -> Store:
        # Opened on the worker thread, which alone may use the connection.
        store = SignallingStore(service.db_path, lambda: 1_000_000.0)
        store.empty = empty
        return store

    worker = service.make_worker(open_store=open_store, idle_wait=1e6)
    finished = threading.Event()
    service.model.on_transcribe.append(finished.set)
    owned_threads.start(worker.run, stop=worker.stop)
    # Submitting only after the worker found the queue empty leaves the wake-up as the only way to the job.
    assert empty.wait(WAIT_SECONDS)

    service.submit(service.media())
    worker.wake()

    assert finished.wait(WAIT_SECONDS)


def test_a_paused_worker_waits_and_resumes_on_its_own(service: Service, owned_threads: ThreadOwner) -> None:
    """While paused the loop checks again after the paused wait, and runs the job once the marker returns."""
    service.marker.unlink()
    job_id = service.submit(service.media())
    worker = service.make_worker(pause_wait=0.01)
    finished = threading.Event()
    service.model.on_transcribe.append(finished.set)
    owned_threads.start(worker.run, stop=worker.stop)
    wait_until(lambda: worker.activity.name is testee.ActivityName.PAUSED, 'the worker to pause')

    service.marker.touch()

    assert finished.wait(WAIT_SECONDS)
    assert service.job(job_id)['state'] in ('running', 'done')


def test_a_store_that_cannot_open_is_fatal(service: Service) -> None:
    """An error before the loop starts reaches the fatal callback once."""
    worker = service.make_worker(open_store=lambda: (_ for _ in ()).throw(sqlite3.OperationalError('locked')))
    worker.run()
    [error] = service.fatal
    assert isinstance(error, sqlite3.OperationalError)
    assert worker.failed


# Review findings on residency, publication, stopping and abandonment


class FreshModelAdapter(FakeAdapter):
    """An adapter that loads a new model object on every load, and records the order of loads and releases."""

    def __init__(self, name: str, events: list[str]) -> None:
        """Share an event log with other adapters."""
        super().__init__(name=name)
        self.events = events
        self.models: list[weakref.ref[FakeModel]] = []

    def load(self, profile: Profile) -> FakeModel:
        """Record whether any earlier model is still alive, then return a new one."""
        gc.collect()
        alive = [ref for ref in self.models + getattr(self, 'peer_models', []) if ref() is not None]
        self.events.append(f'load {profile.name} with {len(alive)} earlier models alive')
        model = FakeModel(SEGMENTS)
        self.models.append(weakref.ref(model))
        return model

    def release(self) -> None:
        """Record the release."""
        self.events.append(f'release {self.name}')


def test_no_earlier_model_is_alive_while_another_profile_loads(service: Service) -> None:
    """Switching profiles, even across engines, drops and releases the old model before loading (DESIGN 5.3)."""
    events: list[str] = []
    first, second = FreshModelAdapter('fake', events), FreshModelAdapter('other', events)
    first.peer_models = second.models  # type: ignore[attr-defined]
    second.peer_models = first.models  # type: ignore[attr-defined]
    other = Profile(name='r', engine='other', model='org/r', revision='r3', precision='float32')
    worker = service.make_worker(profiles={PROFILE.name: PROFILE, OTHER_PROFILE.name: OTHER_PROFILE, 'r': other},
                                 adapters={'fake': first, 'other': second})
    for name, profile in [('one.wav', 'p'), ('two.wav', 'q'), ('three.wav', 'r')]:
        service.submit(service.media(name), profile=profile)
        worker.step(service.store)

    assert events == [
        'load p with 0 earlier models alive',
        'release fake', 'load q with 0 earlier models alive',
        'release fake', 'load r with 0 earlier models alive',
    ]


def test_a_stop_while_the_queue_is_read_starts_no_job(service: Service, tmp_path: Path) -> None:
    """A job selected after shutdown began stays queued for the next start (DESIGN 5.9)."""
    job_id = service.submit(service.media())
    worker = service.make_worker()

    class StoppingStore(Store):
        def next_job(self) -> QueuedJob | None:
            job = super().next_job()
            worker.stop()
            return job

    with closing(StoppingStore(service.db_path, lambda: 1_000_000.0)) as store:
        assert worker.step(store) is testee.Step.IDLE

    assert service.job(job_id)['state'] == 'queued'
    assert service.adapter.loads == []


def test_a_symlink_that_appears_at_the_srt_path_during_inference_is_refused(service: Service, tmp_path: Path) -> None:
    """Containment is checked again before publication, so nothing is written outside the root (DESIGN 5.2)."""
    outside = tmp_path / 'outside'
    outside.mkdir()
    job_id = service.submit(service.media())
    service.model.on_transcribe.append(lambda: (service.root / 'video.srt').symlink_to(outside / 'video.srt'))

    service.worker.step(service.store)

    assert service.failure(job_id) == ('publication', 'filesystem')
    assert list(outside.iterdir()) == []


def test_an_abandoned_job_records_nothing(service: Service) -> None:
    """Inference that returns after abandonment publishes nothing and records nothing (DESIGN 5.9)."""
    job_id = service.submit(service.media())
    service.model.on_transcribe.append(service.worker.abandon)

    service.worker.step(service.store)

    assert not (service.root / 'video.srt').exists()
    job = service.job(job_id)
    assert job['phase'] == 'transcription'  # it never entered publication
    assert (job['state'], job['failure']) == ('running', None)
    assert service.store.list_failures() == []


def test_a_decoding_failure_with_the_marker_gone_fails_filesystem(service: Service) -> None:
    """When the share vanished before the first chunk reached a marker check, the failure is `filesystem`."""
    job_id = service.submit(service.media())

    def unmounted(*args: object) -> object:
        service.marker.unlink()
        raise DecodingError(Path('video.wav'), 1, 'No such file or directory')

    with patch.object(testee, 'run_pipeline', unmounted):
        service.worker.step(service.store)

    assert service.failure(job_id) == ('decoding', 'filesystem')
    failure = service.job(job_id)['failure']
    assert isinstance(failure, dict) and 'staging marker was missing' in str(failure['message'])


def test_a_decoding_failure_with_the_marker_present_stays_decoding(service: Service) -> None:
    """With the share in place, an undecodable file remains a `decoding` failure."""
    job_id = service.submit(service.media())
    with patch.object(testee, 'run_pipeline', side_effect=DecodingError(Path('v'), 1, 'invalid data')):
        service.worker.step(service.store)
    assert service.failure(job_id) == ('decoding', 'decoding')


@pytest.mark.parametrize('at', ['decoding', 'loading'])
def test_an_abandoned_job_stops_at_its_next_step(service: Service, at: str) -> None:
    """Abandonment during decoding stops before the load, and during a load before transcription.

    During decoding, shutdown also kills the decoder; here decoding finishes, and the load is refused.
    """
    job_id = service.submit(service.media())
    if at == 'decoding':
        worker = service.make_worker(marker_check_interval=0.0)
        real_resolve = testee.resolve_job_paths

        def resolve_then_abandon(root: Path, relative: object) -> object:
            paths = real_resolve(root, relative)  # type: ignore[arg-type]
            worker.abandon()
            return paths

        with patch.object(testee, 'resolve_job_paths', resolve_then_abandon):
            worker.step(service.store)
        assert service.adapter.loads == []
    else:
        worker = service.make_worker()
        service.adapter.on_load.append(worker.abandon)
        worker.step(service.store)
        assert service.model.calls == []
    job = service.job(job_id)
    assert (job['state'], job['failure']) == ('running', None)


def test_share_loss_during_inference_fails_filesystem_not_missing(service: Service) -> None:
    """With the marker and the media gone after inference, the share is reported lost (DESIGN 5.10)."""
    job_id = service.submit(service.media())
    service.model.on_transcribe.append(lambda: (service.marker.unlink(), (service.root / 'video.wav').unlink()))

    service.worker.step(service.store)

    assert service.failure(job_id) == ('publication', 'filesystem')
