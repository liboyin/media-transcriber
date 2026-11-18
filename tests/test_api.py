"""Tests for the HTTP API and the service's composition, with fake adapters (DESIGN 5.1, 5.3, 5.8, 5.11, section 6).

The application's lifespan starts a real worker thread on test-owned staging and database. Each test leaves
the client's context, which stops and joins that thread, and the fixture asserts that it has ended.
"""

import builtins
import logging
import os
import sqlite3
import threading
import wave
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from conftest import ProcessExits
from fake_engine import FakeAdapter, FakeModel
from fastapi.testclient import TestClient

import media_transcriber.api as testee
from media_transcriber import faster_whisper_adapter, mlx_whisper_adapter
from media_transcriber.config import ProfileSet, Settings
from media_transcriber.engine import Profile, Segment
from media_transcriber.staging import MARKER_NAME
from media_transcriber.store import Store
from media_transcriber.whisper_languages import WHISPER_LANGUAGES

PROFILE = Profile(name='p', engine='fake', model='org/model', revision='r1', precision='float32')
OTHER = Profile(name='q', engine='fake', model='org/other', revision='r2', precision='float16', decoding={'beam': 5})
WAIT_SECONDS = 10.0
STALE_LIMIT = 120.0
WORKER = 'transcriber-worker'


class ManualClock:
    """A monotonic clock that only the test advances."""

    def __init__(self) -> None:
        """Start at a fixed time."""
        self.now = 1000.0

    def __call__(self) -> float:
        """Return the current time."""
        return self.now


@dataclass
class Harness:
    """Test-owned storage, fakes and settings for one application."""

    tmp_path: Path
    adapter: FakeAdapter = field(default_factory=lambda: FakeAdapter(model=FakeModel([Segment(0.0, 0.1, 'hi')])))
    clock: ManualClock = field(default_factory=ManualClock)
    sleeps: list[float] = field(default_factory=list)
    exits: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Create the staging root with its marker."""
        self.root = self.tmp_path / 'staging'
        self.root.mkdir()
        (self.root / MARKER_NAME).touch()
        self.settings = Settings(
            staging_dir=self.root, database=self.tmp_path / 'db.sqlite3', weights_dir=self.tmp_path / 'weights',
            profiles=ProfileSet({PROFILE.name: PROFILE, OTHER.name: OTHER}, PROFILE.name),
            shutdown_grace=WAIT_SECONDS, staging_health_max_age=STALE_LIMIT, idle_wait=1e6, pause_wait=1e6,
        )

    def components(self, settings: Settings) -> testee.Components:
        """Build the fake components that the lifespan composes."""
        return testee.Components(
            adapters={'fake': self.adapter}, monotonic=self.clock, sleep=self.sleeps.append,
            exit_process=self.exits.append,
        )

    def media(self, name: str = 'video.wav') -> str:
        """Write a short WAV clip into staging and return its relative path."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), 'wb') as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(np.full(1600, 1000, dtype='<i2').tobytes())
        return name


@dataclass
class Api:
    """A running application and its harness."""

    client: TestClient
    harness: Harness

    def runtime(self) -> Any:
        """Return the application's runtime."""
        return self.client.app.state.runtime  # type: ignore[attr-defined]

    def wait_until(self, condition: Callable[[], bool], description: str) -> None:
        """Poll a condition that the worker brings about, failing after a bound."""
        pause = threading.Event()
        for _ in range(int(WAIT_SECONDS / 0.01)):
            if condition():
                return
            pause.wait(0.01)
        raise AssertionError(f'timed out waiting for {description}')

    def wait_for_first_check(self) -> None:
        """Wait until the worker has published its first marker check."""
        self.wait_until(lambda: self.runtime().worker.staging_check is not None, 'the first marker check')

    def job(self, job_id: str) -> dict[str, Any]:
        """Return one job."""
        response = self.client.get(f'/jobs/{job_id}')
        assert response.status_code == 200
        body: dict[str, Any] = response.json()
        return body

    def finished(self, job_id: str) -> dict[str, Any]:
        """Wait until a job has ended and return it."""
        self.wait_until(lambda: self.job(job_id)['state'] in ('done', 'failed'), 'the job to end')
        return self.job(job_id)

    def submit(self, path: str, **body: object) -> dict[str, Any]:
        """Submit a job that must be accepted, and return it."""
        response = self.client.post('/jobs', json={'path': path, 'language': 'en-GB', **body})
        assert response.status_code == 202, response.text
        job: dict[str, Any] = response.json()
        return job

    def health(self) -> tuple[int, dict[str, Any]]:
        """Return the status and body of `/health`."""
        response = self.client.get('/health')
        return response.status_code, response.json()


def run_api(harness: Harness, prepare: Callable[[Harness], None] = lambda harness: None) -> Iterator[Api]:
    """Run the application on a harness until the test ends, then require that its worker thread has ended."""
    prepare(harness)
    app = testee.create_app(harness.settings, harness.components)
    with TestClient(app) as client:
        api = Api(client, harness)
        thread = api.runtime().worker_thread
        try:
            yield api
        finally:
            harness.adapter.model.on_transcribe.clear()
    thread.join(WAIT_SECONDS)
    assert not thread.is_alive()


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    """Provide test-owned storage and fakes."""
    return Harness(tmp_path)


@pytest.fixture
def api(harness: Harness) -> Iterator[Api]:
    """Provide a running application whose worker has made its first marker check."""
    for running in run_api(harness):
        running.wait_for_first_check()
        yield running


# Submission


def test_submit_accepts_a_job_that_the_worker_then_transcribes(api: Api) -> None:
    """A valid submission returns 202 with a queued job, which the worker runs to `done` (DESIGN 5.1, 5.5)."""
    job = api.submit(api.harness.media('a/video.wav'))
    assert (job['state'], job['path'], job['language'], job['engine_language'], job['profile']) == (
        'queued', 'a/video.wav', 'en-GB', 'en', 'p')
    done = api.finished(job['id'])
    assert (done['state'], done['outcome'], done['srt_path']) == ('done', 'transcribed', 'a/video.srt')
    assert (api.harness.root / 'a/video.srt').read_text() == '1\n00:00:00,000 --> 00:00:00,100\nhi\n'


def test_submit_normalizes_the_path_and_runs_the_named_profile(api: Api) -> None:
    """The stored path is normalized, and a named profile overrides the default."""
    job = api.submit('a//b/./video.wav', profile='q', language='zh-Hant-TW')
    assert (job['path'], job['profile'], job['engine_language']) == ('a/b/video.wav', 'q', 'zh')


@pytest.mark.parametrize('body', [
    pytest.param({'path': '', 'language': 'en'}, id='empty-path'),
    pytest.param({'path': '/abs/video.wav', 'language': 'en'}, id='absolute'),
    pytest.param({'path': '../video.wav', 'language': 'en'}, id='parent'),
    pytest.param({'path': 'a/.hidden/video.wav', 'language': 'en'}, id='hidden'),
    pytest.param({'path': 'a\\video.wav', 'language': 'en'}, id='backslash'),
    pytest.param({'path': 'video.wav', 'language': 'en_US'}, id='malformed-tag'),
    pytest.param({'path': 'video.wav', 'language': 'tlh'}, id='unsupported-language'),
    pytest.param({'path': 'video.wav', 'language': 'en', 'profile': 'nope'}, id='unknown-profile'),
    pytest.param({'path': 'video.wav', 'language': 'en', 'priority': 1}, id='extra-field'),
    pytest.param({'path': 'video.wav'}, id='no-language'),
])
def test_submit_refuses_an_invalid_request_without_creating_a_job(api: Api, body: dict[str, object]) -> None:
    """Each invalid submission returns 422, and no job exists afterwards (DESIGN 5.1)."""
    response = api.client.post('/jobs', json=body)
    assert response.status_code == 422, response.text
    assert api.client.get('/jobs').json() == {'jobs': []}


def test_a_profile_whose_engine_has_no_adapter_is_refused(harness: Harness) -> None:
    """A profile whose engine the deployment cannot load refuses submissions."""
    orphan = Profile(name='orphan', engine='gone', model='m', revision='r', precision='float32')
    harness.settings = replace(harness.settings, profiles=ProfileSet({orphan.name: orphan}, orphan.name))
    for running in run_api(harness):
        response = running.client.post('/jobs', json={'path': 'video.wav', 'language': 'en'})
        assert response.status_code == 422
        assert "does not support the language 'en'" in response.text


# Reading jobs and profiles


def test_jobs_are_listed_in_submission_order_and_unknown_jobs_are_404(api: Api) -> None:
    """`GET /jobs` lists every retained job; an unknown ID is 404 (section 6)."""
    first, second = api.submit('one.wav'), api.submit('two.wav')
    assert [job['id'] for job in api.client.get('/jobs').json()['jobs']] == [first['id'], second['id']]
    assert api.client.get('/jobs/unknown').status_code == 404


def test_profiles_lists_settings_default_and_languages(api: Api) -> None:
    """`GET /profiles` reports each profile, which one is the default, and its engine's languages (DESIGN 5.3)."""
    profiles = api.client.get('/profiles').json()['profiles']
    languages = sorted(WHISPER_LANGUAGES)
    assert profiles == [
        {'name': 'p', 'engine': 'fake', 'model': 'org/model', 'revision': 'r1', 'precision': 'float32',
         'decoding': {}, 'default': True, 'languages': languages},
        {'name': 'q', 'engine': 'fake', 'model': 'org/other', 'revision': 'r2', 'precision': 'float16',
         'decoding': {'beam': 5}, 'default': False, 'languages': languages},
    ]
    assert api.harness.adapter.loads == []


def test_profiles_of_an_engine_without_adapter_list_no_languages(harness: Harness) -> None:
    """A profile whose engine has no adapter is still listed, with no languages."""
    orphan = Profile(name='orphan', engine='gone', model='m', revision='r', precision='float32')
    harness.settings = replace(harness.settings, profiles=ProfileSet({orphan.name: orphan}, orphan.name))
    for running in run_api(harness):
        [listed] = running.client.get('/profiles').json()['profiles']
        assert (listed['name'], listed['languages']) == ('orphan', [])


# Failures


def test_failures_are_listed_embedded_and_cleared(api: Api) -> None:
    """A failed job's failure is listed and embedded; clearing one or all removes them (DESIGN 5.8)."""
    first = api.finished(api.submit('absent-1.wav')['id'])
    second = api.finished(api.submit('absent-2.wav')['id'])
    assert (first['failure']['phase'], first['failure']['category']) == ('staging', 'missing')
    failures = api.client.get('/failures').json()['failures']
    assert failures == [first['failure'], second['failure']]

    cleared = api.client.delete(f'/failures/{first["failure"]["id"]}')
    assert cleared.json() == first['failure']
    assert api.client.delete(f'/failures/{first["failure"]["id"]}').status_code == 404
    assert api.job(first['id'])['failure'] is None

    assert api.client.delete('/failures').json() == {'failures': [second['failure']]}
    assert api.client.get('/failures').json() == {'failures': []}


# Health


def test_health_is_200_with_the_full_report_when_everything_works(api: Api) -> None:
    """A healthy service reports its database, staging, worker, profiles and engines (DESIGN 5.11)."""
    status, body = api.health()
    assert status == 200, body
    assert body['healthy'] is True
    assert body['database'] == {'ok': True, 'error': None}
    assert body['staging'] == {'ok': True, 'marker_present': True, 'checked_seconds_ago': 0.0, 'paused': False,
                               'stale': False}
    assert body['worker'] == {'alive': True, 'activity': 'idle', 'job_id': None, 'touches_share': True,
                              'activity_seconds': 0.0, 'progress_seconds_ago': 0.0, 'resident_profile': None}
    assert (body['profiles'], body['default_profile']) == (['p', 'q'], 'p')
    assert body['engines'] == [{'name': 'fake', 'version': None, 'accelerator_available': True}]


def test_health_reports_a_stale_check_while_the_worker_is_idle(api: Api) -> None:
    """An idle worker touches the share, so a marker check older than the limit is unhealthy (M17)."""
    api.harness.clock.now += STALE_LIMIT + 1
    status, body = api.health()
    assert status == 503
    assert body['staging']['stale'] is True and body['staging']['ok'] is False
    assert body['staging']['marker_present'] is True


def test_health_does_not_call_a_long_transcription_stale(api: Api) -> None:
    """During transcription the share is not in use, so an old check stays healthy (M17)."""
    entered, release = threading.Event(), threading.Event()
    api.harness.adapter.model.on_transcribe.append(lambda: (entered.set(), release.wait(WAIT_SECONDS)))
    job = api.submit(api.harness.media())
    try:
        assert entered.wait(WAIT_SECONDS)
        api.harness.clock.now += STALE_LIMIT * 10
        status, body = api.health()
        assert status == 200, body
        assert body['worker']['activity'] == 'transcribing'
        assert (body['worker']['job_id'], body['worker']['touches_share']) == (job['id'], False)
        assert body['worker']['activity_seconds'] == STALE_LIMIT * 10
        assert body['worker']['resident_profile'] == 'p'
        assert body['staging']['checked_seconds_ago'] > STALE_LIMIT
    finally:
        release.set()
    assert api.finished(job['id'])['state'] == 'done'


def test_health_reports_a_paused_worker(harness: Harness) -> None:
    """A missing marker pauses the worker, and health reports it as 503 (DESIGN 5.10)."""
    for running in run_api(harness, lambda harness: (harness.root / MARKER_NAME).unlink()):
        running.wait_for_first_check()
        status, body = running.health()
        assert status == 503
        assert body['staging'] == {'ok': False, 'marker_present': False, 'checked_seconds_ago': 0.0, 'paused': True,
                                   'stale': False}
        assert body['worker']['activity'] == 'paused'


def test_health_reports_a_database_that_cannot_be_written(api: Api) -> None:
    """A failing database check is unhealthy, with its error."""
    with patch.object(Store, 'check', side_effect=sqlite3.OperationalError('database is locked')):
        status, body = api.health()
    assert status == 503
    assert body['database'] == {'ok': False, 'error': 'database is locked'}


def test_health_reports_a_database_with_another_schema_version(api: Api) -> None:
    """A store that cannot open because of a schema mismatch is a 503 with its error, not a 500."""
    with patch.object(Store, 'check', side_effect=RuntimeError('schema version 9, expected 1')):
        status, body = api.health()
    assert status == 503
    assert body['database'] == {'ok': False, 'error': 'schema version 9, expected 1'}


def test_health_reports_a_database_check_that_does_not_finish(api: Api) -> None:
    """A database check that outlasts its bound is unhealthy instead of stalling the response."""
    with patch.object(testee, '_DATABASE_CHECK_SECONDS', 0.0):
        status, body = api.health()
    assert status == 503
    assert body['database'] == {'ok': False, 'error': 'TimeoutError'}


def test_a_failed_worker_is_unhealthy_and_ends_the_process(harness: Harness) -> None:
    """An error that ends the worker leads to the pause and the exit with status 1 (DESIGN 5.9)."""
    with patch.object(Store, 'recover', side_effect=sqlite3.OperationalError('disk I/O error')):
        for running in run_api(harness):
            running.wait_until(lambda: harness.exits == [1], 'the fatal exit')
            status, body = running.health()
            assert status == 503
            assert body['worker']['alive'] is False
    assert harness.sleeps == [harness.settings.fatal_pause]


@pytest.mark.parametrize('cause', ['unavailable', 'raises'])
def test_an_unavailable_accelerator_is_unhealthy(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, cause: str,
) -> None:
    """An unavailable accelerator, or a failed check of one, is logged and makes health 503 (DESIGN 5.11)."""
    if cause == 'unavailable':
        harness.adapter.accelerator = False
    else:
        harness.adapter.accelerator_available = lambda: (_ for _ in ()).throw(RuntimeError('driver'))  # type: ignore[method-assign]
    # The expected error log is checked here and kept from pytest's report, so that a failure below is the
    # only output; the review's mutation tool reads a reported ERROR line as an errored test.
    messages: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())  # type: ignore[method-assign]
    monkeypatch.setattr(testee.logger, 'propagate', False)
    monkeypatch.setattr(testee.logger, 'handlers', [handler])
    for running in run_api(harness):
        running.wait_for_first_check()
        expected = 'Checking the accelerator of fake failed' if cause == 'raises' else 'accelerator of fake is not'
        assert any(expected in message for message in messages), messages
        status, body = running.health()
        assert status == 503
        assert body['engines'] == [{'name': 'fake', 'version': None, 'accelerator_available': False}]


def test_an_engine_without_adapter_is_reported_unavailable(harness: Harness) -> None:
    """A profile's engine with no adapter has no accelerator either."""
    orphan = Profile(name='orphan', engine='gone', model='m', revision='r', precision='float32')
    harness.settings = replace(harness.settings, profiles=ProfileSet({orphan.name: orphan}, orphan.name))
    for running in run_api(harness):
        assert running.health()[1]['engines'] == [{'name': 'gone', 'version': None, 'accelerator_available': False}]


def test_engine_versions_come_from_the_installed_distribution() -> None:
    """An installed engine reports its version; an absent one reports null."""
    assert testee._engine_version('faster-whisper') is not None
    assert testee._engine_version('not-an-installed-engine') is None


# Isolation from the share (DESIGN 5.1)


def test_no_request_touches_the_share(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every filesystem access under the staging root comes from the worker thread, never a request (DESIGN 5.1).

    The probe records Python-level stat, listing and open calls. ffmpeg's own reads are a child process.
    """
    media = harness.media('kept.wav')  # written by the test itself, before the probe records
    accesses: list[tuple[str, str]] = []
    root = str(harness.root)

    def recording[F: Callable[..., Any]](function: F) -> F:
        def wrapper(path: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(path, str | os.PathLike) and os.fspath(path).startswith(root):
                accesses.append((threading.current_thread().name, os.fspath(path)))
            return function(path, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    for module, name in [(os, 'stat'), (os, 'lstat'), (os, 'scandir'), (os, 'listdir'), (os, 'open'),
                         (os, 'access'), (builtins, 'open')]:
        monkeypatch.setattr(module, name, recording(getattr(module, name)))

    for running in run_api(harness):
        running.wait_for_first_check()
        done = running.finished(running.submit(media)['id'])
        failed = running.finished(running.submit('absent.wav')['id'])
        running.client.get('/jobs')
        running.client.get(f'/jobs/{done["id"]}')
        running.client.get('/profiles')
        running.client.get('/failures')
        running.client.get('/health')
        running.client.delete(f'/failures/{failed["failure"]["id"]}')
        running.client.delete('/failures')

    # The worker's accesses show that the probe sees real staging calls.
    assert {thread for thread, _ in accesses} == {WORKER}


# Composition


def test_default_components_offer_both_engines_on_the_weights_directory(harness: Harness) -> None:
    """The real composition has an adapter for each engine, each reading the configured weights directory."""
    adapters = testee.default_components(harness.settings).adapters
    assert sorted(adapters) == [faster_whisper_adapter.ENGINE, mlx_whisper_adapter.ENGINE]
    assert all(adapter.weights_dir == harness.settings.weights_dir for adapter in adapters.values())  # type: ignore[attr-defined]


def test_the_fatal_exit_happens_even_when_the_pause_fails(harness: Harness) -> None:
    """Nothing that the pause raises can keep a process without its worker alive (DESIGN 5.9)."""
    exits: list[int] = []

    def failing_sleep(seconds: float) -> None:
        raise RuntimeError('interrupted pause')

    runtime = testee._Runtime(harness.settings, testee.Components(
        adapters={'fake': harness.adapter}, sleep=failing_sleep, exit_process=exits.append))
    with pytest.raises(RuntimeError, match='interrupted pause'):
        runtime._fatal(sqlite3.OperationalError('broken'))
    assert exits == [1]
    runtime._executor.shutdown()


def test_the_default_exit_ends_the_process_at_once(process_exits: ProcessExits) -> None:
    """The real exit calls `os._exit`, which the test suite refuses and records."""
    with pytest.raises(AssertionError, match='status 1'):
        testee._exit_now(1)
    assert process_exits.take() == [1]


def test_shutdown_abandons_a_job_that_outlasts_the_grace(harness: Harness, caplog: pytest.LogCaptureFixture) -> None:
    """After the grace, shutdown kills decoders and abandons the job, which records nothing more (DESIGN 5.9)."""
    harness.settings = replace(harness.settings, shutdown_grace=0.05)
    entered, release = threading.Event(), threading.Event()
    harness.adapter.model.on_transcribe.append(lambda: (entered.set(), release.wait(WAIT_SECONDS)))
    app = testee.create_app(harness.settings, harness.components)
    kills: list[int] = []
    try:
        with patch.object(testee, 'kill_decoders', lambda: kills.append(1) or 0), TestClient(app) as client:
            thread = app.state.runtime.worker_thread
            response = client.post('/jobs', json={'path': harness.media(), 'language': 'en'})
            assert response.status_code == 202
            assert entered.wait(WAIT_SECONDS)
        assert 'did not stop within the shutdown grace' in caplog.text
        assert kills == [1]
        assert thread.is_alive()
    finally:
        release.set()
    thread.join(WAIT_SECONDS)
    assert not thread.is_alive()
    # The late inference stopped before publication, and nothing was recorded: recovery will repeat it.
    assert not (harness.root / 'video.srt').exists()
    with Store(harness.settings.database, lambda: 0.0) as store:
        job = store.get_job(str(response.json()['id']))
    assert job is not None and (job['state'], job['failure']) == ('running', None)
