"""The HTTP API's response models, which publish every field in the OpenAPI document (DESIGN section 6).

They describe what the store and `/health` already return, and do not reshape it. Extra fields are
forbidden, so a field added to a record without a model here fails loudly instead of going
undocumented. The design follows `youtube-collector`'s responses (assumption A2).
"""

from pydantic import BaseModel, ConfigDict, Field

from media_transcriber.store import Category, JobState, Outcome, Phase
from media_transcriber.worker import ActivityName

__all__ = ['FailureList', 'FailureResponse', 'Health', 'JobList', 'JobResponse', 'ProfileList']


class _Response(BaseModel):
    """A response object whose fields are exactly those declared."""

    model_config = ConfigDict(extra='forbid')


class FailureResponse(_Response):
    """A record that a job did not succeed, kept until cleared or superseded (DESIGN 5.8)."""

    id: str
    created_at: str
    job_id: str
    path: str = Field(description='The media path relative to the staging root.')
    profile: str
    phase: Phase
    category: Category
    message: str = Field(description='The exception type and its raw text.')


class _Durations(_Response):
    """The seconds a job spent in each step. Null means the step did not run (DESIGN 5.7)."""

    queue_wait: float | None
    model_load: float | None
    decoding: float | None
    transcription: float | None = Field(description='Includes the rejection check and SRT serialization.')
    publication: float | None


class JobResponse(_Response):
    """One submission (DESIGN 5.7)."""

    id: str
    path: str = Field(description='The media path relative to the staging root.')
    language: str = Field(description='The BCP 47 tag that the request stated.')
    engine_language: str = Field(description="The code that the profile's engine received.")
    profile: str
    state: JobState
    phase: Phase | None = Field(description='The step a running job is in, or the step where it ended.')
    outcome: Outcome | None = Field(description='Set when the job is done.')
    srt_path: str | None = Field(description='The SRT relative to the staging root, set when the job is done.')
    durations: _Durations
    created_at: str
    started_at: str | None
    finished_at: str | None
    updated_at: str
    failure: FailureResponse | None = Field(description='The uncleared failure of a failed job.')


class JobList(_Response):
    """Every retained job, in submission order (DESIGN 5.8)."""

    jobs: list[JobResponse]


class FailureList(_Response):
    """Failures, oldest first (DESIGN 5.8)."""

    failures: list[FailureResponse]


class _Profile(_Response):
    """One engine profile of the deployment (DESIGN 5.3)."""

    name: str
    engine: str
    model: str
    revision: str
    precision: str
    decoding: dict[str, str | int | float | bool]
    default: bool = Field(description='Whether a request without a profile runs this one.')
    languages: list[str] = Field(description="The engine's language codes, sorted.")


class ProfileList(_Response):
    """The deployment's profiles, in the configured order."""

    profiles: list[_Profile]


class _DatabaseHealth(_Response):
    """Whether the database is readable and writable."""

    ok: bool
    error: str | None


class _StagingHealth(_Response):
    """The worker's most recent marker check (DESIGN 5.10, 5.11)."""

    ok: bool
    marker_present: bool | None = Field(description='Null before the first check.')
    checked_seconds_ago: float | None = Field(description='Null before the first check.')
    paused: bool = Field(description='Whether work is paused because the marker is missing.')
    stale: bool = Field(
        description='Whether the check is older than the limit while the worker is idle or touches the share (M17).',
    )


class _WorkerHealth(_Response):
    """The worker thread and what it is doing (DESIGN 5.11)."""

    alive: bool = Field(description='Whether the thread runs and its run has not failed.')
    activity: ActivityName
    job_id: str | None
    touches_share: bool
    activity_seconds: float = Field(description='How long the current activity has lasted, for information.')
    progress_seconds_ago: float = Field(description='How long ago the worker last reported progress.')
    resident_profile: str | None


class _EngineHealth(_Response):
    """An engine that a configured profile uses."""

    name: str
    version: str | None = Field(description='The installed library version, or null when not installed.')
    accelerator_available: bool = Field(description='Checked once at startup.')


class Health(_Response):
    """Service health. 200 and 503 carry the same body (DESIGN 5.11)."""

    healthy: bool
    database: _DatabaseHealth
    staging: _StagingHealth
    worker: _WorkerHealth
    platform: str
    profiles: list[str]
    default_profile: str
    engines: list[_EngineHealth]
