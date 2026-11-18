"""The worker thread: one job at a time with one resident model (DESIGN 5.5, 5.9, 5.10, 5.11).

The worker is the only thread that touches the staging share. Before each job it checks the
staging marker, and while the marker is missing it pauses without changing any record. A job runs
the steps of DESIGN 5.5 through the staging layer and `run_pipeline`. Its outcome, or its failure
with a category, is recorded in one store write. The worker records each phase in the store before
it enters it, so that recovery can fail a job in the phase where it stopped.

`/health` reads two values that the worker publishes, each an immutable object replaced whole: the
most recent marker check, and the current activity with its progress (DESIGN 5.11, M17, M18).
During decoding and publication the worker checks the marker again between chunks. Progress alone
never refreshes the check, because a successful read or write does not show that the marker is
still present.

An error that escapes the handling of a job is the worker's own fault, not the job's, and it ends
the worker's run. Every store error is such a fault, also one raised while a job runs. The worker
then reports the error to a fatal callback, which ends the process (DESIGN 5.9). The design follows
`youtube-collector`'s worker (assumption A2).
"""

import enum
import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath

from media_transcriber.audio import DecodingError, EmptyAudioError
from media_transcriber.engine import EngineAdapter, Profile, TranscriptionModel
from media_transcriber.pipeline import run_pipeline
from media_transcriber.rejection import RejectedTranscriptionError
from media_transcriber.staging import (
    MARKER_NAME,
    MediaNotFoundError,
    PathOutsideRootError,
    StagingUnavailableError,
    publish_srt,
    require_marker,
    resolve_job_paths,
    srt_exists,
)
from media_transcriber.store import Category, Durations, JobResult, NewFailure, Outcome, Phase, QueuedJob, Store

__all__ = [
    'Activity', 'ActivityName', 'StagingCheck', 'Step', 'UnknownProfileError', 'Worker', 'failure_for',
    'staging_stale',
]

logger = logging.getLogger(__name__)


class ActivityName(StrEnum):
    """What the worker is doing (DESIGN 5.11)."""

    IDLE = 'idle'
    PAUSED = 'paused'
    STAGING = 'staging'
    DECODING = 'decoding'
    LOADING = 'loading'
    TRANSCRIBING = 'transcribing'
    PUBLISHING = 'publishing'


_SHARE_ACTIVITIES = frozenset({
    ActivityName.IDLE, ActivityName.PAUSED, ActivityName.STAGING, ActivityName.DECODING, ActivityName.PUBLISHING,
})
"""The activities that touch the share (DESIGN 5.11). Loading and transcribing work in memory."""


@dataclass(frozen=True)
class Activity:
    """The worker's current activity, published for `/health` (DESIGN 5.11).

    Attributes:
        name: What the worker is doing.
        job_id: The job it serves, or None.
        started_at: When the activity started, on the worker's monotonic clock.
        progress_at: When the worker last reported progress, on the same clock.
    """

    name: ActivityName
    job_id: str | None
    started_at: float
    progress_at: float

    @property
    def touches_share(self) -> bool:
        """Whether the activity reads or writes the staging share, so a stalled share can block it."""
        return self.name in _SHARE_ACTIVITIES


@dataclass(frozen=True)
class StagingCheck:
    """The most recent check of the staging marker, published for `/health` (DESIGN 5.11).

    Attributes:
        marker_present: Whether the marker was present, and so whether work may proceed.
        checked_at: When the check finished, on the worker's monotonic clock.
    """

    marker_present: bool
    checked_at: float


def staging_stale(check: StagingCheck | None, activity: Activity, now: float, limit: float) -> bool:
    """Return whether the last marker check is too old to vouch for the share (DESIGN 5.11, M17).

    Staleness applies only while the worker is idle or in an activity that touches the share. During
    loading and transcription the share is not in use, so a long activity is not a stall.

    Args:
        check: The most recent marker check, or None before the first.
        activity: The worker's current activity.
        now: The current time on the worker's monotonic clock.
        limit: The age in seconds beyond which a check is stale.
    """
    if not activity.touches_share:
        return False
    return check is None or now - check.checked_at > limit


class Step(enum.Enum):
    """What one worker step did."""

    PAUSED = 'paused'
    IDLE = 'idle'
    PROCESSED = 'processed'


class UnknownProfileError(LookupError):
    """A queued job names a profile or an engine that the deployment no longer configures."""


class _StoreFault(Exception):
    """Carries a store error raised while a job runs, so that it ends the worker instead of failing the job."""


def failure_for(phase: Phase, error: BaseException) -> NewFailure:
    """Map an error in a phase to the failure that the job records (DESIGN 5.8).

    The specific staging errors come before the generic `OSError`: `MediaNotFoundError` is a
    `FileNotFoundError`, but a missing ffmpeg binary, or a missing SRT directory during publication,
    raises a plain one, and those mean something else.

    Args:
        phase: The phase that the job was in.
        error: The error that ended the job.

    Returns:
        The failure. A rejection is reported in the phase `rejection`, which runs inside transcription.
    """
    category = Category.INTERNAL
    if isinstance(error, RejectedTranscriptionError):
        phase, category = Phase.REJECTION, Category.REJECTED
    elif isinstance(error, MediaNotFoundError):
        category = Category.MISSING
    elif isinstance(error, StagingUnavailableError):
        category = Category.FILESYSTEM
    elif phase is Phase.MODEL:
        category = Category.MODEL
    elif isinstance(error, (DecodingError, EmptyAudioError)):
        category = Category.DECODING
    elif phase in (Phase.STAGING, Phase.PUBLICATION) and isinstance(error, (OSError, PathOutsideRootError)):
        category = Category.FILESYSTEM
    text = str(error)
    message = f'{type(error).__name__}: {text}' if text else type(error).__name__
    return NewFailure(phase, category, message)


class _JobRun:
    """One job's progress through its phases, with the time spent in each."""

    def __init__(self, job: QueuedJob, now: float) -> None:
        """Start the job in the phase `staging`."""
        self.job = job
        self.phase = Phase.STAGING
        self._entered = now
        self._spent: dict[Phase, float] = {}

    def enter(self, phase: Phase, now: float) -> None:
        """Close the current phase's span and enter another."""
        self._spent[self.phase] = now - self._entered
        self.phase, self._entered = phase, now

    def durations(self, now: float) -> Durations:
        """Return the time spent in each phase, closing the current one at `now`."""
        spent = {**self._spent, self.phase: now - self._entered}
        return Durations(
            model_load=spent.get(Phase.MODEL), decoding=spent.get(Phase.DECODING),
            transcription=spent.get(Phase.TRANSCRIPTION), publication=spent.get(Phase.PUBLICATION),
        )


class Worker:
    """The single thread that processes jobs and touches the staging share.

    Args:
        open_store: Opens the worker's own store. Called on the worker thread.
        staging_root: The staging root.
        profiles: The deployment's profiles by name.
        adapters: The engine adapters by engine name.
        monotonic: A monotonic clock in seconds, for activities, marker checks and durations.
        on_fatal: Called once on the worker thread with the error that ended `run`. The service's
            callback ends the process (DESIGN 5.9). A test's records.
        marker_check_interval: The least time in seconds between marker checks during decoding.
        idle_wait: How long to wait for a wake-up when no job is queued. It must stay below the
            staleness limit of `/health`, so that an idle worker's check stays fresh.
        pause_wait: How long to wait before checking the marker again while paused.
    """

    def __init__(
        self, *, open_store: Callable[[], Store], staging_root: Path, profiles: Mapping[str, Profile],
        adapters: Mapping[str, EngineAdapter], monotonic: Callable[[], float],
        on_fatal: Callable[[BaseException], None], marker_check_interval: float = 10.0, idle_wait: float = 30.0,
        pause_wait: float = 5.0,
    ) -> None:
        """Store the collaborators. Nothing runs until `run`."""
        self._open_store = open_store
        self._root = staging_root
        self._profiles = profiles
        self._adapters = adapters
        self._monotonic = monotonic
        self._on_fatal = on_fatal
        self._marker_check_interval = marker_check_interval
        self._idle_wait = idle_wait
        self._pause_wait = pause_wait
        self._failed = False
        self._paused = False
        self._resident: tuple[str, str, TranscriptionModel] | None = None
        """The resident profile's name, its engine, and its loaded model."""
        self._staging_check: StagingCheck | None = None
        now = monotonic()
        self._activity = Activity(ActivityName.IDLE, None, now, now)
        self._stopping = threading.Event()
        self._abandoned = threading.Event()
        self._wakeup = threading.Event()

    @property
    def failed(self) -> bool:
        """Whether `run` ended on an error. Set before the fatal callback runs, and never cleared."""
        return self._failed

    @property
    def activity(self) -> Activity:
        """The current activity. Safe to read from any thread."""
        return self._activity

    @property
    def staging_check(self) -> StagingCheck | None:
        """The most recent marker check, or None before the first. Safe to read from any thread."""
        return self._staging_check

    @property
    def resident_profile(self) -> str | None:
        """The name of the profile whose model is resident, or None. Safe to read from any thread."""
        resident = self._resident
        return None if resident is None else resident[0]

    def wake(self) -> None:
        """Wake the worker early, for example after a submission. Safe to call from any thread."""
        self._wakeup.set()

    def stop(self) -> None:
        """Ask the worker to stop taking jobs. The current job runs to its end first (DESIGN 5.9)."""
        self._stopping.set()
        self._wakeup.set()

    def abandon(self) -> None:
        """Give up the current job after the shutdown grace, so that restart recovery repeats it (DESIGN 5.9).

        The worker stops taking jobs and records nothing more for the current one, which stays
        `running`. The caller may then kill the job's decoder, whose failure must not be recorded
        as the job's outcome. Safe to call from any thread.
        """
        self._abandoned.set()
        self.stop()

    def run(self) -> None:
        """Recover interrupted jobs, then take jobs one by one until `stop`.

        Job failures are recorded by `step` and never reach here. Anything else that escapes, from
        opening the store, recovery or a step, ends the loop, which would otherwise leave a job
        `running` while the process looks healthy. The worker then marks itself failed and passes
        the error to `on_fatal` exactly once (DESIGN 5.9).
        """
        try:
            with self._open_store() as store:
                store.recover()
                while not self._stopping.is_set():
                    step = self.step(store)
                    if step is Step.PAUSED:
                        self._wait(self._pause_wait)
                    elif step is Step.IDLE:
                        self._wait(self._idle_wait)
        except BaseException as error:
            cause = error.__cause__ if isinstance(error, _StoreFault) and error.__cause__ is not None else error
            self._failed = True
            logger.critical('The worker cannot continue', exc_info=cause)
            self._on_fatal(cause)

    def _wait(self, timeout: float) -> None:
        """Wait for a wake-up, a stop or the timeout."""
        self._wakeup.wait(timeout)
        self._wakeup.clear()

    def step(self, store: Store) -> Step:
        """Take and process at most one job.

        Nothing starts while the staging marker is missing, and no record changes (DESIGN 5.10).
        """
        if not self._check_marker():
            if not self._paused:
                logger.warning('The staging marker %s is missing; pausing work', self._root / MARKER_NAME)
                self._paused = True
            self._set_activity(ActivityName.PAUSED, None)
            return Step.PAUSED
        if self._paused:
            logger.warning('The staging marker is present again; resuming work')
            self._paused = False
        job = store.next_job()
        # A stop that arrives while the queue is read leaves the job queued: the shutdown grace is
        # for a job already in progress, not for one that has yet to start (DESIGN 5.9).
        if job is None or self._stopping.is_set():
            self._set_activity(ActivityName.IDLE, None)
            return Step.IDLE
        self._process(store, job)
        return Step.PROCESSED

    def _process(self, store: Store, job: QueuedJob) -> None:
        """Run one job and record its outcome or its failure in one write."""
        store.start_job(job.id)
        run = _JobRun(job, self._monotonic())
        self._set_activity(ActivityName.STAGING, job.id)
        logger.info('Job %s started: %s with profile %s', job.id, job.path, job.profile)
        try:
            result = self._run_job(store, run)
        except _StoreFault:
            raise
        except Exception as error:
            failure = failure_for(run.phase, error)
            # ffmpeg can fail on a vanished share before the first chunk reaches a marker check.
            if failure.category is Category.DECODING and not self._check_marker():
                failure = replace(failure, category=Category.FILESYSTEM,
                                  message=f'{failure.message} (the staging marker was missing afterwards)')
            logger.warning('Job %s failed in phase %s (%s): %s', job.id, failure.phase, failure.category,
                           failure.message)
            result = JobResult(None, failure, durations=run.durations(self._monotonic()))
        if self._abandoned.is_set():
            logger.warning('Job %s was abandoned at shutdown; restart recovery will repeat it', job.id)
            return
        store.finish_job(job.id, result)
        logger.info('Job %s ended: %s', job.id, result.outcome or 'failed')
        self._set_activity(ActivityName.IDLE, None)

    def _run_job(self, store: Store, run: _JobRun) -> JobResult:
        """Run the steps of DESIGN 5.5 after the marker check, and return the `done` result.

        Raises:
            _StoreFault: If recording a phase fails.
            Exception: Whatever a step raises, which fails the job.
        """
        relative = PurePosixPath(run.job.path)
        paths = resolve_job_paths(self._root, relative)
        srt_path = str(relative.with_suffix('.srt'))
        if srt_exists(paths):
            return JobResult(Outcome.EXISTING, srt_path=srt_path, durations=run.durations(self._monotonic()))
        self._enter(store, run, Phase.DECODING, ActivityName.DECODING)
        result = run_pipeline(
            paths.media, lambda: self._load(store, run), run.job.engine_language, self._decoding_checkpoint,
            self._monotonic,
        )
        self._enter(store, run, Phase.PUBLICATION, ActivityName.PUBLISHING)
        # The share may have changed during inference, so containment and the media are checked again.
        paths = resolve_job_paths(self._root, relative)
        published = publish_srt(paths, result.srt_text, self._publication_checkpoint)
        outcome = Outcome.TRANSCRIBED if published else Outcome.EXISTING
        return JobResult(outcome, srt_path=srt_path, durations=run.durations(self._monotonic()))

    def _load(self, store: Store, run: _JobRun) -> TranscriptionModel:
        """Return the job's model, loading its profile only if it is not resident (DESIGN 5.3, M25).

        Before another profile loads, the worker drops every reference to the resident model and
        asks its adapter to release what it holds, so two models never coexist. A failed load
        leaves no model resident.

        Raises:
            UnknownProfileError: If the deployment no longer configures the job's profile or its engine.
            Exception: Whatever the adapter raises while loading.
        """
        self._enter(store, run, Phase.MODEL, ActivityName.LOADING)
        resident = self._resident
        if resident is None or resident[0] != run.job.profile:
            profile = self._profiles.get(run.job.profile)
            if profile is None:
                raise UnknownProfileError(f'The deployment does not configure the profile {run.job.profile!r}')
            adapter = self._adapters.get(profile.engine)
            if adapter is None:
                raise UnknownProfileError(f'The deployment has no adapter for the engine {profile.engine!r}')
            resident = None  # this frame must not keep the old model alive during the load
            self._release_resident()
            logger.info('Loading the model of profile %s', profile.name)
            resident = (profile.name, profile.engine, adapter.load(profile))
            self._resident = resident
        self._enter(store, run, Phase.TRANSCRIPTION, ActivityName.TRANSCRIBING)
        return resident[2]

    def _release_resident(self) -> None:
        """Drop the resident model and let its adapter release what the engine library still holds."""
        resident, self._resident = self._resident, None
        if resident is None:
            return
        engine = resident[1]
        del resident
        # A model became resident through this adapter, and the map of adapters never changes.
        self._adapters[engine].release()

    def _enter(self, store: Store, run: _JobRun, phase: Phase, activity: ActivityName) -> None:
        """Record that the job enters a phase, then publish the matching activity.

        Raises:
            _StoreFault: If the store cannot record the phase. The job's handler re-raises it.
        """
        try:
            store.set_phase(run.job.id, phase)
        except Exception as error:
            raise _StoreFault(f'Recording phase {phase} of job {run.job.id} failed') from error
        run.enter(phase, self._monotonic())
        self._set_activity(activity, run.job.id)

    def _decoding_checkpoint(self) -> None:
        """Report progress, and check the marker again once the last check is older than the interval (M18).

        Raises:
            StagingUnavailableError: If the marker is missing, which fails the job `filesystem`.
        """
        self._progress()
        check = self._staging_check
        if check is None or self._monotonic() - check.checked_at >= self._marker_check_interval:
            self._require_marker()

    def _publication_checkpoint(self) -> None:
        """Report progress and check the marker, before, during and at the end of publication (DESIGN 5.10).

        Raises:
            StagingUnavailableError: If the marker is missing, so nothing is published.
        """
        self._progress()
        self._require_marker()

    def _require_marker(self) -> None:
        """Check the marker and raise if it is missing.

        Raises:
            StagingUnavailableError: If the marker is missing or the root cannot be read.
        """
        if not self._check_marker():
            raise StagingUnavailableError(f'The staging marker {self._root / MARKER_NAME} is missing')

    def _check_marker(self) -> bool:
        """Check the marker, publish the result for `/health`, and return it."""
        try:
            require_marker(self._root)
        except StagingUnavailableError:
            present = False
        else:
            present = True
        self._staging_check = StagingCheck(present, self._monotonic())
        return present

    def _set_activity(self, name: ActivityName, job_id: str | None) -> None:
        """Publish a new activity, keeping the start of one that continues."""
        current = self._activity
        if (current.name, current.job_id) != (name, job_id):
            now = self._monotonic()
            self._activity = Activity(name, job_id, now, now)

    def _progress(self) -> None:
        """Publish that the current activity made progress."""
        self._activity = replace(self._activity, progress_at=self._monotonic())
