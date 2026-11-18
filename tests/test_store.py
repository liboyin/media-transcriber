"""Tests for the SQLite store of jobs and failures (DESIGN 5.7, 5.8, 5.9)."""

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pytest

import media_transcriber.store as testee

RETENTION = 100.0
START = 1_000_000.0


class Clock:
    """A manually advanced epoch clock."""

    def __init__(self, now: float = START) -> None:
        """Start at a fixed epoch time."""
        self.now = now

    def __call__(self) -> float:
        """Return the current time."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward."""
        self.now += seconds


@pytest.fixture
def clock() -> Clock:
    """Provide a test-owned clock."""
    return Clock()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Provide an initialized database file owned by the test."""
    path = tmp_path / 'transcriber.sqlite3'
    testee.initialize(path)
    return path


@pytest.fixture
def store(db_path: Path, clock: Clock) -> Iterator[testee.Store]:
    """Provide a store on the test's database, closed afterwards."""
    with testee.Store(db_path, clock) as opened:
        yield opened


def submit(store: testee.Store, path: str = 'a/video.mp4', retention: float = RETENTION) -> dict[str, object]:
    """Submit a job for a path with fixed language and profile."""
    return store.submit(path=path, language='zh-CN', engine_language='zh', profile='p', retention_seconds=retention)


def count_rows(db_path: Path, table: str) -> int:
    """Count the committed rows of a table through a separate connection."""
    with closing(sqlite3.connect(db_path)) as connection:
        count: int = connection.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
        return count


def fail(store: testee.Store, job_id: str, phase: testee.Phase = testee.Phase.DECODING) -> None:
    """Start a queued job and end it failed with a `decoding` failure."""
    store.start_job(job_id)
    failure = testee.NewFailure(phase, testee.Category.DECODING, 'DecodingError: bad')
    store.finish_job(job_id, testee.JobResult(None, failure))


def finish_done(store: testee.Store, job_id: str) -> None:
    """Start a queued job and end it `done` with the outcome `transcribed`."""
    store.start_job(job_id)
    store.finish_job(job_id, testee.JobResult(testee.Outcome.TRANSCRIBED, srt_path='a/video.srt'))


def test_initialize_creates_the_schema_in_wal_mode_and_is_idempotent(tmp_path: Path) -> None:
    """A new file gets the current schema version and WAL mode; initializing again keeps its rows."""
    path = tmp_path / 'db.sqlite3'
    testee.initialize(path)
    with testee.Store(path, Clock()) as store:
        submit(store)
    testee.initialize(path)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute('PRAGMA user_version').fetchone()[0] == testee.SCHEMA_VERSION
        assert connection.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
    assert count_rows(path, 'jobs') == 1


def test_initialize_refuses_an_unknown_schema_version(tmp_path: Path) -> None:
    """A database from another schema version is never migrated or overwritten (assumption A1)."""
    path = tmp_path / 'db.sqlite3'
    with closing(sqlite3.connect(path)) as connection:
        connection.execute('PRAGMA user_version = 99')
    with pytest.raises(RuntimeError, match='Unknown database schema version 99'):
        testee.initialize(path)


def test_a_store_refuses_a_database_with_another_schema_version(tmp_path: Path) -> None:
    """Opening a store checks the version, and closes the connection it refuses."""
    path = tmp_path / 'db.sqlite3'
    with closing(sqlite3.connect(path)) as connection:
        connection.execute('PRAGMA user_version = 7')
    with pytest.raises(RuntimeError, match='schema version 7'):
        testee.Store(path, Clock())


def test_job_result_needs_exactly_one_of_an_outcome_and_a_failure() -> None:
    """A result is either done with an outcome or failed with a failure, never both or neither."""
    failure = testee.NewFailure(testee.Phase.MODEL, testee.Category.MODEL, 'x')
    with pytest.raises(ValueError, match='either done'):
        testee.JobResult(None)
    with pytest.raises(ValueError, match='either done'):
        testee.JobResult(testee.Outcome.EXISTING, failure)


def test_check_succeeds_and_writes_nothing(store: testee.Store) -> None:
    """The health check succeeds on a working database and changes no row."""
    job = submit(store)
    store.check()
    assert store.list_jobs() == [job]


def test_check_fails_while_another_connection_holds_the_write_lock(db_path: Path, clock: Clock) -> None:
    """A database that cannot be locked for writing fails the check."""
    with patch.object(testee, '_BUSY_TIMEOUT_SECONDS', 0.0), testee.Store(db_path, clock) as store, \
            closing(sqlite3.connect(db_path, autocommit=True)) as other:
        other.execute('BEGIN IMMEDIATE')
        with pytest.raises(sqlite3.OperationalError, match='locked'):
            store.check()
        other.execute('ROLLBACK')
        store.check()


def test_submit_returns_a_queued_job_with_its_request(store: testee.Store, clock: Clock) -> None:
    """A new job carries its path, both language forms and its profile, and no outcome yet (DESIGN 5.4, 5.7)."""
    job = submit(store)
    assert job == {
        'id': job['id'], 'path': 'a/video.mp4', 'language': 'zh-CN', 'engine_language': 'zh', 'profile': 'p',
        'state': 'queued', 'phase': None, 'outcome': None, 'srt_path': None,
        'durations': {'queue_wait': None, 'model_load': None, 'decoding': None, 'transcription': None,
                      'publication': None},
        'created_at': '1970-01-12T13:46:40+00:00', 'started_at': None, 'finished_at': None,
        'updated_at': '1970-01-12T13:46:40+00:00', 'failure': None,
    }
    assert store.get_job(str(job['id'])) == job


def test_every_submission_creates_a_new_job(store: testee.Store) -> None:
    """Submitting the same path twice creates two jobs, listed in submission order (DESIGN 5.1)."""
    first, second = submit(store), submit(store)
    assert first['id'] != second['id']
    assert [job['id'] for job in store.list_jobs()] == [first['id'], second['id']]


def test_get_job_returns_none_for_an_unknown_job(store: testee.Store) -> None:
    """An unknown ID reads as None, which the API turns into 404."""
    assert store.get_job('unknown') is None


def test_submit_deletes_only_finished_jobs_past_the_retention(store: testee.Store, clock: Clock) -> None:
    """Expired finished jobs go, unfinished jobs of any age stay, and failures outlive their job (DESIGN 5.8)."""
    expired_done, expired_failed, old_queued, old_running = (submit(store)['id'] for _ in range(4))
    finish_done(store, str(expired_done))
    fail(store, str(expired_failed))
    store.start_job(str(old_running))
    clock.advance(RETENTION - 1)
    young = submit(store, 'b/young.mp4')['id']
    finish_done(store, str(young))
    clock.advance(1)

    newest = submit(store)['id']

    assert [job['id'] for job in store.list_jobs()] == [old_queued, old_running, young, newest]
    [failure] = store.list_failures()
    assert failure['job_id'] == expired_failed


def test_next_job_returns_the_oldest_queued_job(store: testee.Store) -> None:
    """The queue is first in, first out, and skips jobs that are not queued."""
    assert store.next_job() is None
    first, second = submit(store, 'one.mp4'), submit(store, 'two.mp4')
    assert store.next_job() == testee.QueuedJob(str(first['id']), 'one.mp4', 'zh', 'p')
    store.start_job(str(first['id']))
    assert store.next_job() == testee.QueuedJob(str(second['id']), 'two.mp4', 'zh', 'p')


def test_start_job_records_the_start_the_phase_and_the_queue_wait(store: testee.Store, clock: Clock) -> None:
    """A started job is running in the phase `staging`, with its wait since submission (DESIGN 5.7)."""
    job_id = str(submit(store)['id'])
    clock.advance(12.5)
    store.start_job(job_id)
    job = store.get_job(job_id)
    assert job is not None
    assert (job['state'], job['phase'], job['started_at']) == ('running', 'staging', '1970-01-12T13:46:52.500000+00:00')
    assert job['durations'] == {'queue_wait': 12.5, 'model_load': None, 'decoding': None, 'transcription': None,
                                'publication': None}


def test_start_job_refuses_a_job_that_is_not_queued(store: testee.Store, db_path: Path) -> None:
    """Starting a running or unknown job raises and changes nothing."""
    job_id = str(submit(store)['id'])
    store.start_job(job_id)
    before = store.get_job(job_id)
    with pytest.raises(LookupError, match='No queued job'):
        store.start_job(job_id)
    with pytest.raises(LookupError):
        store.start_job('unknown')
    assert store.get_job(job_id) == before


def test_set_phase_records_the_phase_of_a_running_job_only(store: testee.Store) -> None:
    """A running job's phase moves on; a queued job has no phase to set."""
    job_id = str(submit(store)['id'])
    with pytest.raises(LookupError, match='No running job'):
        store.set_phase(job_id, testee.Phase.DECODING)
    store.start_job(job_id)
    store.set_phase(job_id, testee.Phase.TRANSCRIPTION)
    job = store.get_job(job_id)
    assert job is not None and job['phase'] == 'transcription'


def test_finish_job_done_records_the_outcome_path_and_durations(store: testee.Store, clock: Clock) -> None:
    """A done job carries its outcome, the SRT path and every duration it measured (DESIGN 5.7)."""
    job_id = str(submit(store)['id'])
    store.start_job(job_id)
    clock.advance(30)
    durations = testee.Durations(model_load=1.0, decoding=2.0, transcription=3.0, publication=4.0)
    store.finish_job(job_id, testee.JobResult(testee.Outcome.TRANSCRIBED, srt_path='a/video.srt',
                                              durations=durations))
    job = store.get_job(job_id)
    assert job is not None
    assert (job['state'], job['outcome'], job['srt_path'], job['failure']) == ('done', 'transcribed',
                                                                               'a/video.srt', None)
    assert job['finished_at'] == '1970-01-12T13:47:10+00:00'
    assert job['durations'] == {'queue_wait': 0.0, 'model_load': 1.0, 'decoding': 2.0, 'transcription': 3.0,
                                'publication': 4.0}


def test_finish_job_failed_records_and_embeds_the_failure(store: testee.Store) -> None:
    """A failed job keeps its failure's phase and embeds the failure, which is also listed (DESIGN 5.8)."""
    job_id = str(submit(store)['id'])
    store.start_job(job_id)
    store.set_phase(job_id, testee.Phase.TRANSCRIPTION)
    failure = testee.NewFailure(testee.Phase.REJECTION, testee.Category.REJECTED, 'RejectedTranscriptionError: x')
    store.finish_job(job_id, testee.JobResult(None, failure, durations=testee.Durations(decoding=2.0)))
    job = store.get_job(job_id)
    assert job is not None
    assert (job['state'], job['phase'], job['outcome']) == ('failed', 'rejection', None)
    assert job['failure'] == {
        'id': job['failure']['id'], 'created_at': '1970-01-12T13:46:40+00:00', 'job_id': job_id,  # type: ignore[index]
        'path': 'a/video.mp4', 'profile': 'p', 'phase': 'rejection', 'category': 'rejected',
        'message': 'RejectedTranscriptionError: x',
    }
    assert store.list_failures() == [job['failure']]
    assert job['durations']['decoding'] == 2.0  # type: ignore[index]


def test_finish_job_refuses_a_job_that_is_not_running(store: testee.Store, db_path: Path) -> None:
    """Ending a queued job raises and records no failure, because the transaction rolls back."""
    job_id = str(submit(store)['id'])
    failure = testee.NewFailure(testee.Phase.MODEL, testee.Category.MODEL, 'x')
    with pytest.raises(LookupError):
        store.finish_job(job_id, testee.JobResult(None, failure))
    with pytest.raises(LookupError):
        store.finish_job(job_id, testee.JobResult(testee.Outcome.EXISTING))
    assert count_rows(db_path, 'failures') == 0
    job = store.get_job(job_id)
    assert job is not None and job['state'] == 'queued'


def test_a_done_job_supersedes_the_failures_of_its_path_only(store: testee.Store) -> None:
    """A later done job for the same path clears its earlier failures; other paths keep theirs (DESIGN 5.8)."""
    failed_same, failed_other = submit(store)['id'], submit(store, 'b/other.mp4')['id']
    fail(store, str(failed_same))
    fail(store, str(failed_other))
    retried = submit(store)['id']
    finish_done(store, str(retried))
    assert [failure['job_id'] for failure in store.list_failures()] == [failed_other]
    job = store.get_job(str(failed_same))
    assert job is not None and job['failure'] is None


def test_a_failed_job_does_not_supersede_earlier_failures(store: testee.Store) -> None:
    """Only a done job supersedes, so two failures for one path are both kept, oldest first."""
    first, second = submit(store)['id'], submit(store)['id']
    fail(store, str(first))
    fail(store, str(second))
    assert [failure['job_id'] for failure in store.list_failures()] == [first, second]


def test_clear_failures_returns_and_deletes_every_failure(store: testee.Store) -> None:
    """Clearing all returns the cleared records oldest first and removes them from their jobs."""
    first, second = submit(store)['id'], submit(store, 'b.mp4')['id']
    fail(store, str(first))
    fail(store, str(second))
    listed = store.list_failures()
    assert store.clear_failures() == listed
    assert store.list_failures() == []
    job = store.get_job(str(first))
    assert job is not None and (job['state'], job['failure']) == ('failed', None)


def test_clear_failure_deletes_one_failure(store: testee.Store) -> None:
    """Clearing one returns it, keeps the others, and reports None the second time."""
    first, second = submit(store)['id'], submit(store, 'b.mp4')['id']
    fail(store, str(first))
    fail(store, str(second))
    target, kept = store.list_failures()
    assert store.clear_failure(str(target['id'])) == target
    assert store.list_failures() == [kept]
    assert store.clear_failure(str(target['id'])) is None


def test_recover_requeues_running_jobs_below_the_cap(store: testee.Store) -> None:
    """A job left running returns to queued without its phase or start, ready to run again (DESIGN 5.9)."""
    job_id = str(submit(store)['id'])
    store.start_job(job_id)
    store.set_phase(job_id, testee.Phase.TRANSCRIPTION)
    store.recover()
    job = store.get_job(job_id)
    assert job is not None
    assert (job['state'], job['phase'], job['started_at'], job['failure']) == ('queued', None, None, None)
    assert store.next_job() is not None


def test_recover_fails_a_job_at_its_third_interruption_in_its_phase(store: testee.Store) -> None:
    """The third interruption fails the job `internal` in the phase it was in, instead of requeueing it."""
    job_id = str(submit(store)['id'])
    for interruption in range(1, testee.MAX_INTERRUPTIONS + 1):
        store.start_job(job_id)
        store.set_phase(job_id, testee.Phase.PUBLICATION)
        store.recover()
        job = store.get_job(job_id)
        assert job is not None
        assert job['state'] == ('queued' if interruption < testee.MAX_INTERRUPTIONS else 'failed')
    assert job['finished_at'] is not None
    assert job['failure'] is not None
    failure = job['failure']
    assert isinstance(failure, dict)
    assert (failure['phase'], failure['category']) == ('publication', 'internal')
    assert 'Interrupted 3 times' in str(failure['message'])
    assert store.next_job() is None


def test_recover_leaves_queued_and_finished_jobs_alone(store: testee.Store) -> None:
    """Only running jobs count an interruption."""
    queued, done = submit(store)['id'], submit(store)['id']
    finish_done(store, str(done))
    before = store.list_jobs()
    store.recover()
    assert store.list_jobs() == before
    assert queued in [job['id'] for job in before]


def deny_once(store: testee.Store, *statements: str) -> None:
    """Make SQLite refuse the next COMMIT or ROLLBACK on the store's connection, once each."""
    remaining = set(statements)

    def authorize(action: int, argument: str | None, *_: object) -> int:
        if action == sqlite3.SQLITE_TRANSACTION and argument in remaining:
            remaining.discard(argument)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    assert store._connection is not None
    store._connection.set_authorizer(authorize)


def test_a_refused_commit_acknowledges_nothing_and_leaves_the_store_usable(store: testee.Store, db_path: Path) -> None:
    """A submission whose COMMIT fails raises, keeps no job, and the next submission works."""
    deny_once(store, 'COMMIT')
    with pytest.raises(sqlite3.DatabaseError, match='not authorized'):
        submit(store)
    assert count_rows(db_path, 'jobs') == 0
    assert submit(store)['state'] == 'queued'


def test_a_transaction_that_sqlite_already_ended_keeps_the_original_error(store: testee.Store) -> None:
    """No ROLLBACK is tried when none is open, so a healthy connection is not replaced."""
    original = store._connection
    with pytest.raises(ValueError, match='body'), store._write() as connection:
        connection.execute('ROLLBACK')
        raise ValueError('body')
    assert store._connection is original
    store.check()


def test_a_failed_rollback_keeps_the_original_error_and_replaces_the_connection(
    store: testee.Store, db_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """When cleanup fails too, the caller sees the first error and the uncommitted work is discarded."""
    original = store._connection
    deny_once(store, 'ROLLBACK')
    with pytest.raises(ValueError, match='body'), store._write() as connection:
        connection.execute(
            "INSERT INTO jobs (id, path, language, engine_language, profile, state, created_at, updated_at)"
            " VALUES ('uncommitted', 'p', 'en', 'en', 'p', 'queued', 0, 0)"
        )
        raise ValueError('body')
    assert 'replacing the database connection' in caplog.text
    assert count_rows(db_path, 'jobs') == 0
    assert submit(store)['state'] == 'queued'
    assert store._connection is not original
    assert original is not None
    with pytest.raises(sqlite3.ProgrammingError):
        original.execute('SELECT 1')


def test_a_replacement_connection_that_cannot_open_is_tried_again(store: testee.Store) -> None:
    """An unavailable database fails that operation only; a later one opens the replacement."""
    deny_once(store, 'ROLLBACK')
    with pytest.raises(ValueError, match='body'), store._write():
        raise ValueError('body')
    with patch.object(testee.sqlite3, 'connect', side_effect=sqlite3.OperationalError('unavailable')), \
            pytest.raises(sqlite3.OperationalError, match='unavailable'):
        store.check()
    store.check()


def test_closing_a_store_twice_is_harmless(db_path: Path, clock: Clock) -> None:
    """A closed store has no connection, and closing it again raises nothing."""
    store = testee.Store(db_path, clock)
    store.close()
    store.close()
    assert store._connection is None
