"""The SQLite store: jobs and failures (DESIGN 5.7, 5.8, 5.9).

SQLite is the source of truth. Every write is one short `BEGIN IMMEDIATE` transaction, and a
result is reported only once it has committed. A finished job stays readable for a retention period
and is deleted by a later submission (DESIGN 5.8). A failure stays until it is cleared, or until a
later job for the same path ends `done`, and it outlives its job.

Each thread opens its own `Store`. sqlite3's thread check stays on, so a connection used from
another thread raises instead of corrupting a transaction. The database is never on the staging
share (DESIGN section 9). The design follows `youtube-collector`'s store (assumption A2).
"""

import logging
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Self

__all__ = [
    'MAX_INTERRUPTIONS', 'SCHEMA_VERSION', 'Category', 'Durations', 'JobResult', 'JobState', 'NewFailure', 'Outcome',
    'Phase', 'QueuedJob', 'Record', 'Store', 'initialize',
]

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_INTERRUPTIONS = 3
"""Recovery fails a job at its third interruption instead of requeueing it, so that a job that ends
the process every time cannot block the queue in a restart loop (DESIGN 5.9)."""

_BUSY_TIMEOUT_SECONDS = 30.0

type Record = dict[str, object]


class JobState(StrEnum):
    """A job's lifecycle state (DESIGN 5.7)."""

    QUEUED = 'queued'
    RUNNING = 'running'
    DONE = 'done'
    FAILED = 'failed'


class Outcome(StrEnum):
    """How a `done` job ended (DESIGN 5.7)."""

    TRANSCRIBED = 'transcribed'
    EXISTING = 'existing'


class Phase(StrEnum):
    """The step of DESIGN 5.5 that a job is in, or that failed (DESIGN 5.8)."""

    STAGING = 'staging'
    MODEL = 'model'
    DECODING = 'decoding'
    TRANSCRIPTION = 'transcription'
    REJECTION = 'rejection'
    PUBLICATION = 'publication'


class Category(StrEnum):
    """What kind of failure happened (DESIGN 5.8)."""

    MISSING = 'missing'
    DECODING = 'decoding'
    REJECTED = 'rejected'
    MODEL = 'model'
    FILESYSTEM = 'filesystem'
    INTERNAL = 'internal'


@dataclass(frozen=True)
class NewFailure:
    """A failure to record. The store adds the job's path and profile.

    Attributes:
        phase: The step that failed.
        category: What kind of failure it was.
        message: The raw error text.
    """

    phase: Phase
    category: Category
    message: str


@dataclass(frozen=True)
class Durations:
    """The durations a job measured, in seconds. None means the step did not run.

    The queue wait is not here: the store measures it when the job starts.
    """

    model_load: float | None = None
    decoding: float | None = None
    transcription: float | None = None
    publication: float | None = None


@dataclass(frozen=True)
class JobResult:
    """How a running job ended: `done` with an outcome, or `failed` with a failure.

    Attributes:
        outcome: The outcome of a `done` job, or None for a failed job.
        failure: The failure of a failed job, or None for a `done` job.
        srt_path: The SRT's path relative to the staging root, for a `done` job.
        durations: What the job measured before it ended.
    """

    outcome: Outcome | None
    failure: NewFailure | None = None
    srt_path: str | None = None
    durations: Durations = field(default_factory=Durations)

    def __post_init__(self) -> None:
        """Require exactly one of an outcome and a failure.

        Raises:
            ValueError: If both or neither are given.
        """
        if (self.outcome is None) == (self.failure is None):
            raise ValueError('A job ends either done with an outcome or failed with a failure')


@dataclass(frozen=True)
class QueuedJob:
    """A queued job, as the worker needs it.

    Attributes:
        id: The job's ID.
        path: The media path relative to the staging root, as `validate_request_path` normalized it.
        engine_language: The language code that the engine receives.
        profile: The name of the profile to run.
    """

    id: str
    path: str
    engine_language: str
    profile: str


_SCHEMA = """
CREATE TABLE jobs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    language TEXT NOT NULL,
    engine_language TEXT NOT NULL,
    profile TEXT NOT NULL,
    state TEXT NOT NULL,
    phase TEXT,
    outcome TEXT,
    srt_path TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    updated_at REAL NOT NULL,
    queue_wait_seconds REAL,
    model_load_seconds REAL,
    decoding_seconds REAL,
    transcription_seconds REAL,
    publication_seconds REAL,
    interruptions INTEGER NOT NULL DEFAULT 0
) STRICT;
CREATE INDEX jobs_by_state ON jobs (state, seq);
CREATE INDEX jobs_by_finished_at ON jobs (finished_at);
CREATE TABLE failures (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    job_id TEXT NOT NULL,
    path TEXT NOT NULL,
    profile TEXT NOT NULL,
    phase TEXT NOT NULL,
    category TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at REAL NOT NULL
) STRICT;
CREATE INDEX failures_by_path ON failures (path);
CREATE INDEX failures_by_job ON failures (job_id);
"""


def initialize(path: Path) -> None:
    """Create the database at `path` if needed, enable WAL mode and check the schema version.

    Call once at startup, before any thread opens a `Store`.

    Raises:
        RuntimeError: If the file holds a schema version this code does not know.
        sqlite3.Error: If the database cannot be opened or written.
    """
    connection = sqlite3.connect(path, timeout=_BUSY_TIMEOUT_SECONDS, autocommit=True)
    try:
        connection.execute('PRAGMA journal_mode = WAL')
        connection.execute('BEGIN IMMEDIATE')
        try:
            version = connection.execute('PRAGMA user_version').fetchone()[0]
            if version == 0:
                connection.executescript(_SCHEMA)
                connection.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
            elif version != SCHEMA_VERSION:
                raise RuntimeError(f'Unknown database schema version {version} in {path}')
        except BaseException:
            connection.execute('ROLLBACK')
            raise
        connection.execute('COMMIT')
    finally:
        connection.close()


def _iso(timestamp: float | None) -> str | None:
    """Format an epoch timestamp as ISO 8601 UTC, or pass None through."""
    return None if timestamp is None else datetime.fromtimestamp(timestamp, UTC).isoformat()


class Store:
    """One thread's connection to the database, with every read and write the service needs.

    Args:
        path: The database file, already set up by `initialize`.
        clock: Returns the current time as epoch seconds. Injected so that tests control it.
    """

    def __init__(self, path: Path, clock: Callable[[], float]) -> None:
        """Open the connection. `initialize` must have created the schema."""
        self._path = path
        self._clock = clock
        self._connection: sqlite3.Connection | None = self._connect()

    def _connect(self) -> sqlite3.Connection:
        """Open and check a connection to the database.

        Raises:
            RuntimeError: If the file holds a different schema version.
            sqlite3.Error: If the database cannot be opened or read.
        """
        connection = sqlite3.connect(self._path, timeout=_BUSY_TIMEOUT_SECONDS, autocommit=True)
        try:
            connection.row_factory = sqlite3.Row
            version = connection.execute('PRAGMA user_version').fetchone()[0]
            if version != SCHEMA_VERSION:
                raise RuntimeError(f'Database {self._path} has schema version {version}, expected {SCHEMA_VERSION}')
        except BaseException:
            connection.close()
            raise
        return connection

    def _connected(self) -> sqlite3.Connection:
        """Return the connection, opening a replacement if a failed cleanup discarded the last one."""
        if self._connection is None:
            self._connection = self._connect()
        return self._connection

    def close(self) -> None:
        """Close the connection, if a failed cleanup has not already discarded it."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> Self:
        """Return the store, which closes on exit."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None,
    ) -> None:
        """Close the connection."""
        self.close()

    # Transactions

    @contextmanager
    def _transaction(self, begin: str) -> Iterator[sqlite3.Connection]:
        """Run the body as one transaction that commits, or ends without leaving the connection in it.

        The caller sees the error that ended the transaction, whether the body or `COMMIT` raised
        it, so a result is reported only once it has committed.

        Args:
            begin: The statement that starts the transaction.
        """
        connection = self._connected()
        connection.execute(begin)
        try:
            yield connection
            connection.execute('COMMIT')
        except BaseException:
            self._end_failed_transaction(connection)
            raise

    def _end_failed_transaction(self, connection: sqlite3.Connection) -> None:
        """Roll back whatever a failure left open, and replace the connection if that fails too.

        SQLite rolls some failures back by itself, so `ROLLBACK` runs only if a transaction is
        still open. A cleanup error is logged instead of raised, because the caller needs the error
        that ended the transaction. Closing discards the open transaction. The next operation opens
        a replacement, since the database may still be unavailable now.
        """
        try:
            if connection.in_transaction:
                connection.execute('ROLLBACK')
        except sqlite3.Error:
            logger.exception('Rolling back a failed transaction failed; replacing the database connection')
            self._connection = None
            connection.close()

    def _write(self) -> AbstractContextManager[sqlite3.Connection]:
        """Run the body as one immediate write transaction."""
        return self._transaction('BEGIN IMMEDIATE')

    def _read(self) -> AbstractContextManager[sqlite3.Connection]:
        """Run the body as one read transaction, so every statement sees the same snapshot."""
        return self._transaction('BEGIN')

    def check(self) -> None:
        """Check that the database is readable and writable, for `/health` (DESIGN 5.11).

        Takes the write lock with `BEGIN IMMEDIATE`, reads a row, and changes nothing.

        Raises:
            sqlite3.Error: If the database cannot be read or locked for writing.
        """
        with self._write() as connection:
            connection.execute('SELECT 1 FROM jobs LIMIT 1').fetchone()

    # Submission

    def submit(
        self, *, path: str, language: str, engine_language: str, profile: str, retention_seconds: float,
    ) -> Record:
        """Create a queued job and return its representation, first deleting expired jobs.

        Only submissions add jobs, so deleting expired jobs here bounds the table without a
        background task. The retention is therefore a lower bound. An unfinished job has no
        `finished_at` and never expires. Failures are kept (DESIGN 5.8).

        Args:
            path: The media path relative to the staging root, already validated.
            language: The BCP 47 tag that the request stated.
            engine_language: The code that the profile's engine receives for it.
            profile: The name of the profile to run.
            retention_seconds: How long a finished job stays readable.

        Returns:
            The new job.
        """
        with self._write() as connection:
            now = self._clock()
            connection.execute('DELETE FROM jobs WHERE finished_at <= ?', (now - retention_seconds,))
            job_id = uuid.uuid4().hex
            connection.execute(
                'INSERT INTO jobs (id, path, language, engine_language, profile, state, created_at, updated_at)'
                ' VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (job_id, path, language, engine_language, profile, JobState.QUEUED, now, now),
            )
            return self._job_record(connection, job_id)

    # Reads

    def get_job(self, job_id: str) -> Record | None:
        """Return a job with its uncleared failure, or None when it is unknown or deleted."""
        with self._read() as connection:
            row = connection.execute('SELECT id FROM jobs WHERE id = ?', (job_id,)).fetchone()
            return None if row is None else self._job_record(connection, job_id)

    def list_jobs(self) -> list[Record]:
        """Return every retained job, in submission order."""
        with self._read() as connection:
            ids = [row['id'] for row in connection.execute('SELECT id FROM jobs ORDER BY seq')]
            return [self._job_record(connection, job_id) for job_id in ids]

    def list_failures(self) -> list[Record]:
        """Return every uncleared failure, oldest first."""
        with self._read() as connection:
            return [_failure_record(row) for row in connection.execute('SELECT * FROM failures ORDER BY seq')]

    @staticmethod
    def _job_record(connection: sqlite3.Connection, job_id: str) -> Record:
        """Build a job's representation, with its uncleared failure or None (DESIGN 5.7)."""
        row = connection.execute('SELECT * FROM jobs WHERE id = ?', (job_id,)).fetchone()
        failure = connection.execute('SELECT * FROM failures WHERE job_id = ? ORDER BY seq', (job_id,)).fetchone()
        return {
            'id': row['id'],
            'path': row['path'],
            'language': row['language'],
            'engine_language': row['engine_language'],
            'profile': row['profile'],
            'state': row['state'],
            'phase': row['phase'],
            'outcome': row['outcome'],
            'srt_path': row['srt_path'],
            'durations': {
                'queue_wait': row['queue_wait_seconds'],
                'model_load': row['model_load_seconds'],
                'decoding': row['decoding_seconds'],
                'transcription': row['transcription_seconds'],
                'publication': row['publication_seconds'],
            },
            'created_at': _iso(row['created_at']),
            'started_at': _iso(row['started_at']),
            'finished_at': _iso(row['finished_at']),
            'updated_at': _iso(row['updated_at']),
            'failure': None if failure is None else _failure_record(failure),
        }

    # Worker writes

    def next_job(self) -> QueuedJob | None:
        """Return the oldest queued job, or None when nothing is queued."""
        with self._read() as connection:
            row = connection.execute(
                "SELECT id, path, engine_language, profile FROM jobs WHERE state = 'queued' ORDER BY seq LIMIT 1",
            ).fetchone()
        return None if row is None else QueuedJob(row['id'], row['path'], row['engine_language'], row['profile'])

    def start_job(self, job_id: str) -> None:
        """Mark a queued job as running in the phase `staging`, and record its queue wait.

        Raises:
            LookupError: If the job is not queued.
        """
        with self._write() as connection:
            now = self._clock()
            cursor = connection.execute(
                'UPDATE jobs SET state = ?, phase = ?, started_at = ?, queue_wait_seconds = ? - created_at,'
                ' updated_at = ? WHERE id = ? AND state = ?',
                (JobState.RUNNING, Phase.STAGING, now, now, now, job_id, JobState.QUEUED),
            )
            if cursor.rowcount != 1:
                raise LookupError(f'No {JobState.QUEUED} job with ID {job_id!r}')

    def set_phase(self, job_id: str, phase: Phase) -> None:
        """Record the phase that a running job enters.

        Raises:
            LookupError: If the job is not running.
        """
        with self._write() as connection:
            self._update(connection, job_id, JobState.RUNNING, self._clock(), phase=phase)

    def finish_job(self, job_id: str, result: JobResult) -> None:
        """Record how a running job ended, with its failure, in one transaction (DESIGN 5.7, 5.8).

        A `done` job supersedes the uncleared failures of earlier jobs for the same path. A failed
        job keeps the phase of its failure.

        Raises:
            LookupError: If the job is not running.
        """
        durations = result.durations
        fields: dict[str, object] = {
            'model_load_seconds': durations.model_load, 'decoding_seconds': durations.decoding,
            'transcription_seconds': durations.transcription, 'publication_seconds': durations.publication,
        }
        with self._write() as connection:
            now = self._clock()
            if result.failure is None:
                self._update(
                    connection, job_id, JobState.RUNNING, now, state=JobState.DONE, outcome=result.outcome,
                    srt_path=result.srt_path, finished_at=now, **fields,
                )
                path = connection.execute('SELECT path FROM jobs WHERE id = ?', (job_id,)).fetchone()[0]
                connection.execute('DELETE FROM failures WHERE path = ?', (path,))
            else:
                self._update(
                    connection, job_id, JobState.RUNNING, now, state=JobState.FAILED, phase=result.failure.phase,
                    finished_at=now, **fields,
                )
                self._insert_failure(connection, job_id, result.failure, now)

    # Failures

    def clear_failures(self) -> list[Record]:
        """Delete every uncleared failure and return the cleared records, oldest first."""
        with self._write() as connection:
            rows = connection.execute('SELECT * FROM failures ORDER BY seq').fetchall()
            connection.execute('DELETE FROM failures')
            return [_failure_record(row) for row in rows]

    def clear_failure(self, failure_id: str) -> Record | None:
        """Delete one failure and return its record, or None when it is unknown or already cleared."""
        with self._write() as connection:
            row = connection.execute('DELETE FROM failures WHERE id = ? RETURNING *', (failure_id,)).fetchone()
            return None if row is None else _failure_record(row)

    # Maintenance

    def recover(self) -> None:
        """Requeue the jobs that a restart or a crash interrupted, up to a cap per job (DESIGN 5.9).

        Each running job counts one more interruption. Below `MAX_INTERRUPTIONS` it returns to
        `queued`. At the cap it fails with an `internal` failure in the phase it was in.
        """
        with self._write() as connection:
            now = self._clock()
            rows = connection.execute(
                "SELECT id, phase, interruptions FROM jobs WHERE state = 'running' ORDER BY seq",
            ).fetchall()
            for row in rows:
                count = row['interruptions'] + 1
                if count < MAX_INTERRUPTIONS:
                    self._update(
                        connection, row['id'], JobState.RUNNING, now, state=JobState.QUEUED, phase=None,
                        started_at=None, interruptions=count,
                    )
                    continue
                logger.warning('Job %s was interrupted %d times; failing it', row['id'], count)
                phase = Phase(row['phase'] or Phase.STAGING)
                self._update(
                    connection, row['id'], JobState.RUNNING, now, state=JobState.FAILED, finished_at=now,
                    interruptions=count,
                )
                message = f'Interrupted {count} times by a restart or a crash of the service; not requeued again'
                self._insert_failure(connection, row['id'], NewFailure(phase, Category.INTERNAL, message), now)

    # Row helpers

    @staticmethod
    def _update(
        connection: sqlite3.Connection, job_id: str, expected: JobState, now: float, **fields: object,
    ) -> None:
        """Set columns of one job in the expected state, stamping `updated_at`.

        Raises:
            LookupError: If no job with that ID is in the expected state.
        """
        assignments = ', '.join(f'{column} = ?' for column in fields)
        cursor = connection.execute(
            f'UPDATE jobs SET {assignments}, updated_at = ? WHERE id = ? AND state = ?',
            (*fields.values(), now, job_id, expected),
        )
        if cursor.rowcount != 1:
            raise LookupError(f'No {expected} job with ID {job_id!r}')

    @staticmethod
    def _insert_failure(connection: sqlite3.Connection, job_id: str, failure: NewFailure, now: float) -> None:
        """Insert a failure with its job's path and profile."""
        job = connection.execute('SELECT path, profile FROM jobs WHERE id = ?', (job_id,)).fetchone()
        connection.execute(
            'INSERT INTO failures (id, job_id, path, profile, phase, category, message, created_at)'
            ' VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (uuid.uuid4().hex, job_id, job['path'], job['profile'], failure.phase, failure.category,
             failure.message, now),
        )


def _failure_record(row: sqlite3.Row) -> Record:
    """Return a failure's representation (DESIGN 5.8)."""
    return {
        'id': row['id'],
        'created_at': _iso(row['created_at']),
        'job_id': row['job_id'],
        'path': row['path'],
        'profile': row['profile'],
        'phase': row['phase'],
        'category': row['category'],
        'message': row['message'],
    }
