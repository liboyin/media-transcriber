"""Tests for the service entry point (DESIGN section 9) and its process lifecycle (DESIGN 5.9).

The lifecycle tests run real service processes through `service_process.py`, with a fake engine, on loopback
and against test-owned storage. A test's SIGKILL stands in for a supervisor's fallback; it does not validate
launchd or Docker.
"""

import http.client
import json
import os
import runpy
import signal
import socket
import subprocess
import sys
import time
import wave
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import network_guard
import numpy as np
import pytest
import service_process

import media_transcriber.__main__ as testee
from media_transcriber.config import SettingsError
from media_transcriber.staging import MARKER_NAME

WAIT_SECONDS = 30.0
SERVICE_SCRIPT = Path(service_process.__file__)
PACKAGE_DIR = Path(testee.__file__).parent
SRT = f'1\n00:00:00,000 --> 00:00:00,100\n{service_process.TEXT}\n'
REQUIRED = {'TRANSCRIBER_DATABASE': 'db.sqlite3', 'TRANSCRIBER_WEIGHTS_DIR': 'weights'}


def test_main_serves_one_process_with_the_configured_address(tmp_path: Path) -> None:
    """The entry point builds the app from the environment and runs a single uvicorn server on it."""
    environ = {**REQUIRED, 'TRANSCRIBER_STAGING_DIR': str(tmp_path), 'TRANSCRIBER_PORT': '9123',
               'TRANSCRIBER_HOST': '127.0.0.1'}
    with patch.object(testee.uvicorn, 'run') as run, patch.object(testee.logging, 'basicConfig'):
        testee.main(environ)
    [app], options = run.call_args
    assert app.title == 'media-transcriber'
    assert (options['host'], options['port']) == ('127.0.0.1', 9123)
    assert options['timeout_graceful_shutdown'] == testee.REQUEST_SHUTDOWN_SECONDS
    assert 'workers' not in options  # one process: the worker and its resident model are in-process


def test_main_fails_without_settings() -> None:
    """A missing staging directory stops the service before it serves."""
    with patch.object(testee.uvicorn, 'run') as run, patch.object(testee.logging, 'basicConfig'), \
            pytest.raises(SettingsError):
        testee.main({})
    run.assert_not_called()


def test_running_the_package_calls_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`python3 -m media_transcriber` runs main with the process environment."""
    monkeypatch.setenv('TRANSCRIBER_STAGING_DIR', str(tmp_path))
    for name, value in REQUIRED.items():
        monkeypatch.setenv(name, str(tmp_path / value))
    # A fresh execution, as `python3 -m` performs; the imported module is restored afterwards.
    monkeypatch.delitem(sys.modules, 'media_transcriber.__main__')
    with patch.object(testee.uvicorn, 'run') as run, patch.object(testee.logging, 'basicConfig'):
        runpy.run_module('media_transcriber', run_name='__main__')
    run.assert_called_once()


def wait_until(condition: Callable[[], bool], description: str) -> None:
    """Wait for a condition that another process brings about, failing the test after a bound."""
    deadline = time.monotonic() + WAIT_SECONDS
    while not condition():
        assert time.monotonic() < deadline, f'timed out waiting for {description}'
        time.sleep(0.01)


class ServiceProcess:
    """One running service process, driven over loopback and by its control directory."""

    def __init__(self, process: subprocess.Popen[bytes], port: int, log: Path) -> None:
        """Remember the child, its port and its combined output."""
        self.process = process
        self.port = port
        self.log = log

    def output(self) -> str:
        """Return everything the process has logged so far."""
        return self.log.read_text()

    def request(self, method: str, path: str, body: object = None) -> tuple[int, Any]:
        """Send one request and return its status and decoded JSON body."""
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=WAIT_SECONDS)
        try:
            connection.request(method, path, None if body is None else json.dumps(body),
                               {'Content-Type': 'application/json'})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def healthy(self) -> bool:
        """Return whether the service answers /health with 200 yet."""
        assert self.process.poll() is None, f'the service exited early:\n{self.output()}'
        try:
            return self.request('GET', '/health')[0] == 200
        except OSError:
            return False

    def submit(self, path: str) -> str:
        """Submit a job and return its ID."""
        status, job = self.request('POST', '/jobs', {'path': path, 'language': 'en'})
        assert status == 202, job
        job_id: str = job['id']
        return job_id

    def job(self, job_id: str) -> dict[str, Any]:
        """Return a job's current representation."""
        status, job = self.request('GET', f'/jobs/{job_id}')
        assert status == 200
        return job  # type: ignore[no-any-return]

    def wait_ended(self, job_id: str) -> dict[str, Any]:
        """Wait for a job to end and return it."""
        wait_until(lambda: self.job(job_id)['state'] in ('done', 'failed'), 'the job to end')
        return self.job(job_id)

    def terminate(self) -> None:
        """Send SIGTERM, as launchd and `docker compose stop` do."""
        self.process.send_signal(signal.SIGTERM)

    def wait_exit(self) -> int:
        """Wait for the process to exit by itself and return its exit status."""
        return self.process.wait(WAIT_SECONDS)


class Lifecycle:
    """The test-owned storage and control files shared by the service processes of one test."""

    def __init__(self, tmp_path: Path) -> None:
        """Create the staging root with its marker, a clip to transcribe, and the control directory."""
        self.tmp_path = tmp_path
        self.staging = tmp_path / 'staging'
        self.staging.mkdir()
        (self.staging / MARKER_NAME).touch()
        with wave.open(str(self.staging / 'clip.wav'), 'wb') as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(np.full(1600, 1000, dtype='<i2').tobytes())
        self.control = tmp_path / 'control'
        self.control.mkdir()
        self.home = tmp_path / 'home'
        self.home.mkdir()
        self.processes: list[subprocess.Popen[bytes]] = []

    def flag(self, name: str) -> Path:
        """Return one control file."""
        return self.control / name

    def srt(self) -> Path:
        """Return the clip's SRT path."""
        return self.staging / 'clip.srt'

    def transcriptions(self) -> int:
        """Return how many transcriptions have started, across every process of the test."""
        path = self.flag('transcriptions')
        return len(path.read_text().splitlines()) if path.exists() else 0

    def refusals(self) -> list[str]:
        """Return the network guard's refusals that any service process of the test has logged."""
        return [line for log in sorted(self.tmp_path.glob('service-*.log')) for line in log.read_text().splitlines()
                if network_guard.MARKER in line]

    def start(self, grace: str = '10') -> ServiceProcess:
        """Start a service process on a free loopback port and wait until it is healthy."""
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        environment = {
            'PATH': os.environ['PATH'], 'HOME': str(self.home), 'TMPDIR': str(self.tmp_path),
            'PYTHONPATH': str(PACKAGE_DIR.parent), 'TRANSCRIBER_STAGING_DIR': str(self.staging),
            'TRANSCRIBER_DATABASE': str(self.tmp_path / 'transcriber.sqlite3'),
            'TRANSCRIBER_WEIGHTS_DIR': str(self.tmp_path / 'weights'), 'TRANSCRIBER_HOST': '127.0.0.1',
            'TRANSCRIBER_PORT': str(port), 'TRANSCRIBER_SHUTDOWN_GRACE': grace,
        }
        log = self.tmp_path / f'service-{len(self.processes)}.log'
        with log.open('wb') as output:
            process = subprocess.Popen(
                [sys.executable, str(SERVICE_SCRIPT), str(self.control)], env=environment, cwd=self.tmp_path,
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            )
        self.processes.append(process)
        service = ServiceProcess(process, port, log)
        wait_until(service.healthy, 'the service to become healthy')
        # A child importing another tree would make every result below meaningless.
        assert f'PACKAGE {PACKAGE_DIR / "__init__.py"}' in service.output()
        return service

    def close(self) -> None:
        """Kill and reap every process, then fail if any survives or tried to reach beyond loopback."""
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.wait(WAIT_SECONDS)
        assert all(process.returncode is not None for process in self.processes)
        refusals = self.refusals()
        assert not refusals, f'A service process tried to reach beyond loopback: {refusals}'


@pytest.fixture
def lifecycle(tmp_path: Path) -> Iterator[Lifecycle]:
    """Provide test-owned storage; teardown kills and reaps every service process, also after a failure."""
    owned = Lifecycle(tmp_path)
    try:
        yield owned
    finally:
        owned.close()


def assert_stopped_gracefully(service: ServiceProcess, exit_status: int) -> None:
    """Check that the application finished shutting down and uvicorn re-raised the signal it handled."""
    assert 'Application shutdown complete' in service.output()
    assert exit_status == -signal.SIGTERM


def test_a_service_process_transcribes_a_job(lifecycle: Lifecycle) -> None:
    """A real process accepts a job, transcribes it with its engine and publishes the SRT."""
    service = lifecycle.start()
    job = service.wait_ended(service.submit('clip.wav'))
    assert (job['state'], job['outcome']) == ('done', 'transcribed')
    assert lifecycle.srt().read_text() == SRT


def test_sigterm_ends_an_idle_service_gracefully(lifecycle: Lifecycle) -> None:
    """With nothing running, SIGTERM completes the application's shutdown and ends the process."""
    service = lifecycle.start()
    service.terminate()
    assert_stopped_gracefully(service, service.wait_exit())
    assert 'did not stop within the shutdown grace' not in service.output()


def test_sigterm_lets_the_current_job_finish_within_the_grace(lifecycle: Lifecycle) -> None:
    """A job that ends within the grace is published and recorded before the process exits (DESIGN 5.9)."""
    lifecycle.flag('hold-transcription').touch()
    service = lifecycle.start()
    job_id = service.submit('clip.wav')
    wait_until(lifecycle.flag('transcription-entered').exists, 'the transcription to start')
    service.terminate()
    wait_until(lambda: 'Shutting down' in service.output(), 'shutdown to begin')
    lifecycle.flag('hold-transcription').unlink()
    assert_stopped_gracefully(service, service.wait_exit())
    assert 'did not stop within the shutdown grace' not in service.output()

    restarted = lifecycle.start()
    job = restarted.job(job_id)
    assert (job['state'], job['outcome']) == ('done', 'transcribed')
    assert lifecycle.srt().read_text() == SRT
    assert lifecycle.transcriptions() == 1


def test_a_job_outlasting_the_grace_is_abandoned_and_redone_after_restart(lifecycle: Lifecycle) -> None:
    """The process exits with the job unfinished and nothing published; restart recovery runs it again."""
    lifecycle.flag('hold-transcription').touch()
    service = lifecycle.start(grace='0.2')
    job_id = service.submit('clip.wav')
    wait_until(lifecycle.flag('transcription-entered').exists, 'the transcription to start')
    service.terminate()
    assert_stopped_gracefully(service, service.wait_exit())
    assert 'did not stop within the shutdown grace' in service.output()
    assert not lifecycle.srt().exists()

    lifecycle.flag('hold-transcription').unlink()
    restarted = lifecycle.start()
    job = restarted.wait_ended(job_id)
    assert (job['state'], job['outcome']) == ('done', 'transcribed')
    assert lifecycle.srt().read_text() == SRT
    assert lifecycle.transcriptions() == 2


def test_a_failed_worker_ends_the_process_with_status_1(lifecycle: Lifecycle) -> None:
    """A store error on the worker thread ends the process after the fatal pause, for a restart (DESIGN 5.9)."""
    service = lifecycle.start()
    lifecycle.flag('fail-store').touch()
    service.submit('clip.wav')
    assert service.wait_exit() == 1
    output = service.output()
    assert 'The worker cannot continue' in output and 'Exiting with status 1' in output
    assert 'Application shutdown complete' not in output
