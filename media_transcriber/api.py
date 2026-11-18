"""The HTTP API and the service's composition (DESIGN 5.1, 5.3, 5.8, 5.9, 5.11 and section 6).

One process runs the HTTP server and the worker thread. Request handlers never touch the staging
share (DESIGN 5.1): the worker owns it, and handlers read the database and the values that the
worker publishes. Database calls run on one dedicated thread with its own connection, so the event
loop never blocks on SQLite. The design follows `youtube-collector`'s API (assumption A2).
"""

import asyncio
import contextlib
import importlib.metadata
import logging
import os
import sqlite3
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from media_transcriber import faster_whisper_adapter, mlx_whisper_adapter
from media_transcriber.config import Settings
from media_transcriber.engine import EngineAdapter
from media_transcriber.languages import is_well_formed
from media_transcriber.responses import FailureList, FailureResponse, Health, JobList, JobResponse, ProfileList
from media_transcriber.staging import InvalidPathError, validate_request_path
from media_transcriber.store import Store, initialize
from media_transcriber.worker import Worker, staging_stale

__all__ = ['Components', 'create_app', 'default_components']

logger = logging.getLogger(__name__)

type Body = dict[str, object]

_DATABASE_CHECK_SECONDS = 5.0


def _exit_now(status: int) -> None:
    """End the process at once with `status`, skipping every shutdown handler (DESIGN 5.9).

    `os._exit` is looked up at each call rather than bound as a default, so that the test suite's
    guard can refuse it.
    """
    os._exit(status)


@dataclass
class Components:
    """The collaborators that the service composes. Tests substitute fakes for the adapters and clocks.

    Attributes:
        adapters: The engine adapters by engine name.
        clock: Wall-clock epoch seconds, for the store.
        monotonic: A monotonic clock, for the worker and `/health`.
        sleep: Waits out the pause before a fatal exit.
        exit_process: Ends the process at once with a status, without any shutdown (DESIGN 5.9).
    """

    adapters: Mapping[str, EngineAdapter]
    clock: Callable[[], float] = field(default=time.time)
    monotonic: Callable[[], float] = field(default=time.monotonic)
    sleep: Callable[[float], None] = field(default=time.sleep)
    exit_process: Callable[[int], object] = field(default=_exit_now)


def default_components(settings: Settings) -> Components:
    """Return the real adapters, both loading from the configured weights directory (M22)."""
    adapters: list[EngineAdapter] = [
        faster_whisper_adapter.FasterWhisperAdapter(settings.weights_dir),
        mlx_whisper_adapter.MlxWhisperAdapter(settings.weights_dir),
    ]
    return Components(adapters={adapter.name: adapter for adapter in adapters})


class _JobRequest(BaseModel):
    """The body of `POST /jobs` (DESIGN 5.1). Every check is syntactic and never touches the share."""

    model_config = ConfigDict(extra='forbid')

    path: str = Field(description='The media file relative to the staging root, with `/` as the separator.')
    language: str = Field(description='The source language as a BCP 47 tag, such as `zh-CN` (DESIGN 5.4).')
    profile: str | None = Field(default=None, description="A profile's name. Absent means the default profile.")

    @field_validator('path')
    @classmethod
    def _relative(cls, path: str) -> str:
        """Reject a path that is empty, absolute, or has a `..` or hidden component; return it normalized."""
        try:
            return str(validate_request_path(path))
        except InvalidPathError as error:
            raise ValueError(str(error)) from None

    @field_validator('language')
    @classmethod
    def _well_formed(cls, language: str) -> str:
        """Reject a tag that is not a well-formed BCP 47 tag."""
        if not is_well_formed(language):
            raise ValueError(f'Not a well-formed BCP 47 language tag: {language!r}')
        return language


def _engine_version(engine: str) -> str | None:
    """Return the installed version of an engine's library, whose distribution has the engine's name."""
    try:
        return importlib.metadata.version(engine)
    except importlib.metadata.PackageNotFoundError:
        return None


class _Runtime:
    """Everything the service starts: the worker thread and the API's database thread."""

    def __init__(self, settings: Settings, components: Components) -> None:
        """Create the database if needed, check each engine's accelerator once, and build the worker.

        Nothing runs until `start`.
        """
        self.settings = settings
        self.components = components
        initialize(settings.database)
        engines = dict.fromkeys(profile.engine for profile in settings.profiles.profiles.values())
        self.accelerators = {engine: self._accelerator(engine) for engine in engines}
        self.worker = Worker(
            open_store=self._open_store, staging_root=settings.staging_dir, profiles=settings.profiles.profiles,
            adapters=components.adapters, monotonic=components.monotonic, on_fatal=self._fatal,
            marker_check_interval=settings.marker_check_interval, idle_wait=settings.idle_wait,
            pause_wait=settings.pause_wait,
        )
        self.worker_thread = threading.Thread(target=self.worker.run, name='transcriber-worker', daemon=True)
        self._local = threading.local()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='api-db')

    def _accelerator(self, engine: str) -> bool:
        """Return whether an engine's accelerator is available, treating a failed check as unavailable."""
        adapter = self.components.adapters.get(engine)
        if adapter is None:
            logger.error('No adapter exists for the engine %s', engine)
            return False
        try:
            available = adapter.accelerator_available()
        except Exception:
            logger.exception('Checking the accelerator of %s failed', engine)
            return False
        if not available:
            logger.error('The accelerator of %s is not available; its jobs will fail', engine)
        return available

    def _fatal(self, error: BaseException) -> None:
        """End the process with status 1 after the fatal pause, for the supervisor to restart it (DESIGN 5.9).

        Runs on the worker thread once its run has ended on `error`. There is no graceful shutdown,
        because the fault is typically a broken database, on which the API's database thread can
        block without a bound. The API keeps serving during the pause, and `/health` reports the
        failed worker. The exit is the contract and the diagnostics are best effort: nothing that
        the pause or a flush raises can prevent it, because a process left alive without its
        worker is never restarted.
        """
        try:
            logger.critical(
                'The worker failed (%s); exiting with status 1 in %s seconds for the supervisor to restart the service',
                type(error).__name__, self.settings.fatal_pause,
            )
            self.components.sleep(self.settings.fatal_pause)
            logger.critical('Exiting with status 1')
            for flush in (*(handler.flush for handler in logging.getLogger().handlers), sys.stdout.flush,
                          sys.stderr.flush):
                with contextlib.suppress(Exception):  # a lost log line must not keep the process alive
                    flush()
        finally:
            self.components.exit_process(1)

    def _open_store(self) -> Store:
        """Open a store for the calling thread."""
        return Store(self.settings.database, self.components.clock)

    def _api_store(self) -> Store:
        """Return the API database thread's store, opening it on first use.

        Opening lazily, inside an operation, makes a failure to open an ordinary `sqlite3.Error` of
        that operation, which `/health` reports as 503, and lets a later call try again.
        """
        store: Store | None = getattr(self._local, 'store', None)
        if store is None:
            store = self._local.store = self._open_store()
        return store

    def _close_api_store(self) -> None:
        """Close the API database thread's store, if it was ever opened."""
        store: Store | None = getattr(self._local, 'store', None)
        if store is not None:
            store.close()
            self._local.store = None

    async def db[T](self, operation: Callable[[Store], T]) -> T:
        """Run a store operation on the API's database thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: operation(self._api_store()))

    def start(self) -> None:
        """Start the worker thread."""
        self.worker_thread.start()

    def stop(self) -> None:
        """Ask the worker to stop, wait up to the shutdown grace for it, then close the API's database thread.

        The grace bounds waiting, not execution. A worker still busy when it ends is abandoned and
        keeps running until the process exits, and restart recovery requeues its job (DESIGN 5.9).
        """
        self.worker.stop()
        self.worker_thread.join(self.settings.shutdown_grace)
        if self.worker_thread.is_alive():
            logger.warning('%s did not stop within the shutdown grace; abandoning it', self.worker_thread.name)
        try:
            self._executor.submit(self._close_api_store).result()
        finally:
            self._executor.shutdown(wait=True)


def create_app(
    settings: Settings, components_factory: Callable[[Settings], Components] = default_components,
) -> FastAPI:
    """Build the FastAPI application. Its lifespan starts and stops the worker thread.

    Args:
        settings: The service settings.
        components_factory: Builds the adapters and clocks at startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Build the runtime and run the worker for the application's lifetime."""
        runtime = _Runtime(settings, components_factory(settings))
        runtime.start()
        app.state.runtime = runtime
        try:
            yield
        finally:
            await asyncio.to_thread(runtime.stop)

    app = FastAPI(
        title='media-transcriber', version='0.0.1', lifespan=lifespan,
        description='Transcribes media on a staging share into SRT subtitles. LAN-only and unauthenticated; '
                    'see DESIGN.md.',
    )

    def runtime(request: Request) -> _Runtime:
        """Return the running service."""
        current: _Runtime = request.app.state.runtime
        return current

    service = Annotated[_Runtime, Depends(runtime)]

    @app.post('/jobs', status_code=status.HTTP_202_ACCEPTED, summary='Submit a transcription job',
              response_model=JobResponse)
    async def submit(body: _JobRequest, rt: service) -> Body:
        """Queue one media file for transcription and wake the worker (DESIGN 5.1).

        422 for an invalid path or tag, an unknown profile, or a language that the profile's engine
        does not support. The share is not touched, so a missing file fails the job later as `missing`.
        """
        profiles = rt.settings.profiles
        name = body.profile if body.profile is not None else profiles.default
        profile = profiles.profiles.get(name)
        if profile is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                                f'Unknown profile {name!r}; known: {list(profiles.profiles)}')
        adapter = rt.components.adapters.get(profile.engine)
        engine_language = None if adapter is None else adapter.engine_language(body.language)
        if engine_language is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                                f'Profile {name!r} does not support the language {body.language!r}')
        job = await rt.db(lambda store: store.submit(
            path=body.path, language=body.language, engine_language=engine_language, profile=name,
            retention_seconds=rt.settings.job_retention,
        ))
        rt.worker.wake()
        return job

    @app.get('/jobs', summary='Every retained job', response_model=JobList)
    async def list_jobs(rt: service) -> Body:
        """Every retained job with its uncleared failure, in submission order (DESIGN 5.8)."""
        return {'jobs': await rt.db(lambda store: store.list_jobs())}

    @app.get('/jobs/{job_id}', summary='One job', response_model=JobResponse)
    async def get_job(job_id: str, rt: service) -> Body:
        """One job with its uncleared failure. Poll this to follow it (DESIGN 5.8).

        404 for an unknown job and for one past its retention.
        """
        job = await rt.db(lambda store: store.get_job(job_id))
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f'Unknown job {job_id}')
        return job

    @app.get('/profiles', summary="The deployment's profiles", response_model=ProfileList)
    async def list_profiles(rt: service) -> Body:
        """Each profile's settings and supported languages, without loading a model (DESIGN 5.3)."""
        profiles = rt.settings.profiles
        listed = []
        for profile in profiles.profiles.values():
            adapter = rt.components.adapters.get(profile.engine)
            languages = [] if adapter is None else sorted(adapter.supported_languages())
            listed.append({
                'name': profile.name, 'engine': profile.engine, 'model': profile.model,
                'revision': profile.revision, 'precision': profile.precision, 'decoding': dict(profile.decoding),
                'default': profile.name == profiles.default, 'languages': languages,
            })
        return {'profiles': listed}

    @app.get('/failures', summary='Uncleared failures', response_model=FailureList)
    async def list_failures(rt: service) -> Body:
        """Every uncleared failure, oldest first (DESIGN 5.8)."""
        return {'failures': await rt.db(lambda store: store.list_failures())}

    @app.delete('/failures', summary='Clear all failures', response_model=FailureList)
    async def clear_failures(rt: service) -> Body:
        """Clear every failure and return the cleared records."""
        return {'failures': await rt.db(lambda store: store.clear_failures())}

    @app.delete('/failures/{failure_id}', summary='Clear one failure', response_model=FailureResponse)
    async def clear_failure(failure_id: str, rt: service) -> Body:
        """Clear one failure and return it. 404 when it is unknown or already cleared."""
        cleared = await rt.db(lambda store: store.clear_failure(failure_id))
        if cleared is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f'Unknown failure {failure_id}')
        return cleared

    @app.get(
        '/health', summary='Service health', response_model=Health,
        responses={503: {'model': Health, 'description': 'Unhealthy; same body.'}},
    )
    async def health(rt: service) -> JSONResponse:
        """Database, staging, worker, profiles and engines, without touching the share (DESIGN 5.11).

        Staging is judged from the worker's last marker check. Its age counts only while the worker
        is idle or in an activity that touches the share (M17). 200 when healthy, 503 otherwise,
        with the same body.
        """
        database_error = None
        try:
            await asyncio.wait_for(rt.db(lambda store: store.check()), _DATABASE_CHECK_SECONDS)
        except (sqlite3.Error, TimeoutError) as error:
            database_error = str(error) or type(error).__name__
        worker = rt.worker
        check, activity = worker.staging_check, worker.activity
        now = rt.components.monotonic()
        stale = staging_stale(check, activity, now, rt.settings.staging_health_max_age)
        staging_ok = check is not None and check.marker_present and not stale
        # A failed worker's thread lives on only to pause before the process exits (DESIGN 5.9).
        alive = rt.worker_thread.is_alive() and not worker.failed
        accelerators_ok = all(rt.accelerators.values())
        healthy = database_error is None and staging_ok and alive and accelerators_ok
        body = {
            'healthy': healthy,
            'database': {'ok': database_error is None, 'error': database_error},
            'staging': {
                'ok': staging_ok, 'marker_present': None if check is None else check.marker_present,
                'checked_seconds_ago': None if check is None else round(now - check.checked_at, 3),
                'paused': check is not None and not check.marker_present, 'stale': stale,
            },
            'worker': {
                'alive': alive, 'activity': activity.name, 'job_id': activity.job_id,
                'touches_share': activity.touches_share,
                'activity_seconds': round(now - activity.started_at, 3),
                'progress_seconds_ago': round(now - activity.progress_at, 3),
                'resident_profile': worker.resident_profile,
            },
            'platform': sys.platform,
            'profiles': list(rt.settings.profiles.profiles),
            'default_profile': rt.settings.profiles.default,
            'engines': [
                {'name': engine, 'version': _engine_version(engine), 'accelerator_available': available}
                for engine, available in rt.accelerators.items()
            ],
        }
        return JSONResponse(body, status_code=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE)

    return app
