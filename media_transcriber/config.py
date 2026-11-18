"""Service settings from the environment, and the deployment's profiles (DESIGN 5.3, section 9, M24).

This is the only module that reads the environment. Every other component takes explicit
parameters, so tests never depend on the process environment.
"""

import math
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from media_transcriber import faster_whisper_adapter, mlx_whisper_adapter
from media_transcriber.engine import Profile

__all__ = ['ENGINES', 'ProfileSet', 'Settings', 'SettingsError', 'baseline_profile', 'load_profiles']

ENGINES = frozenset({faster_whisper_adapter.ENGINE, mlx_whisper_adapter.ENGINE})
"""The engines that a profile may name (DESIGN section 8)."""

_PROFILE_FIELDS = ('name', 'engine', 'model', 'revision', 'precision')
_PROFILE_KEYS = frozenset({*_PROFILE_FIELDS, 'decoding'})
_FILE_KEYS = frozenset({'default', 'profiles'})


class SettingsError(ValueError):
    """An environment variable or the profiles file is missing or invalid. The message names it."""


@dataclass(frozen=True)
class ProfileSet:
    """The profiles a deployment offers, in the configured order, and its default (DESIGN 5.3).

    Attributes:
        profiles: The profiles by name.
        default: The name of the profile that a request without one runs.
    """

    profiles: Mapping[str, Profile]
    default: str


def baseline_profile(platform: str) -> Profile:
    """Return the platform's baseline profile: MLX Whisper on macOS, faster-whisper elsewhere (M24)."""
    return mlx_whisper_adapter.BASELINE_PROFILE if platform == 'darwin' else faster_whisper_adapter.BASELINE_PROFILE


def load_profiles(path: Path | None, platform: str) -> ProfileSet:
    """Read the profiles file, or offer the platform's baseline profile without one (M24).

    The file holds `[[profiles]]` tables with `name`, `engine`, `model`, `revision` and
    `precision`, and an optional `decoding` table of scalar settings. A top-level `default` names
    the default profile. Without it, the first profile is the default. The adapter checks the
    precision and the decoding settings when it loads the profile.

    Args:
        path: The TOML file, or None for the baseline.
        platform: `sys.platform`, which selects the baseline.

    Raises:
        SettingsError: If the file cannot be read or parsed, has an unknown key, lacks a field, has
            a duplicate name or an unknown engine, or names a default that it does not define.
    """
    if path is None:
        profile = baseline_profile(platform)
        return ProfileSet({profile.name: profile}, profile.name)
    try:
        with path.open('rb') as file:
            document = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SettingsError(f'The profiles file {path} cannot be read: {error}') from None
    unknown = set(document) - _FILE_KEYS
    if unknown:
        raise SettingsError(f'The profiles file {path} has unknown keys: {sorted(unknown)}')
    tables = document.get('profiles')
    if not isinstance(tables, list) or not tables:
        raise SettingsError(f'The profiles file {path} must define at least one [[profiles]] table')
    profiles: dict[str, Profile] = {}
    for index, table in enumerate(tables):
        profile = _profile(path, index, table)
        if profile.name in profiles:
            raise SettingsError(f'The profiles file {path} defines the profile {profile.name!r} twice')
        profiles[profile.name] = profile
    default = document.get('default', next(iter(profiles)))
    if not isinstance(default, str) or default not in profiles:
        raise SettingsError(f'The profiles file {path} names the default {default!r}, which it does not define')
    return ProfileSet(profiles, default)


def _profile(path: Path, index: int, table: object) -> Profile:
    """Build one profile from its TOML table.

    Raises:
        SettingsError: If the table is malformed.
    """
    where = f'Profile {index + 1} in {path}'
    if not isinstance(table, dict):
        raise SettingsError(f'{where} is not a table')
    unknown = set(table) - _PROFILE_KEYS
    if unknown:
        raise SettingsError(f'{where} has unknown keys: {sorted(unknown)}')
    values = {}
    for name in _PROFILE_FIELDS:
        value = table.get(name)
        if not isinstance(value, str) or not value:
            raise SettingsError(f'{where} needs `{name}` as a non-empty string')
        values[name] = value
    if values['engine'] not in ENGINES:
        raise SettingsError(f'{where} names the unknown engine {values["engine"]!r}; known: {sorted(ENGINES)}')
    decoding = table.get('decoding', {})
    if not isinstance(decoding, dict) or not all(
        isinstance(value, str | int | float | bool) for value in decoding.values()
    ):
        raise SettingsError(f'{where} needs `decoding` as a table of strings, numbers and booleans')
    return Profile(**values, decoding=decoding)


@dataclass(frozen=True)
class Settings:
    """The service's settings, with the defaults of the T8 boundary.

    Attributes:
        staging_dir: The staging root (`TRANSCRIBER_STAGING_DIR`, required).
        database: The SQLite file, never on the share (`TRANSCRIBER_DATABASE`, required).
        weights_dir: The weights directory, laid out as a Hugging Face cache (`TRANSCRIBER_WEIGHTS_DIR`,
            required, M22).
        profiles: The profiles from `TRANSCRIBER_PROFILES_FILE`, or the platform's baseline (M24).
        host: The address the HTTP server binds (`TRANSCRIBER_HOST`).
        port: The TCP port (`TRANSCRIBER_PORT`).
        job_retention: Seconds a finished job stays readable (`TRANSCRIBER_JOB_RETENTION_DAYS`, in days).
        shutdown_grace: How long shutdown waits for the current job before abandoning it, which does
            not stop it (`TRANSCRIBER_SHUTDOWN_GRACE`, DESIGN 5.9).
        staging_health_max_age: The age beyond which a marker check is stale during an activity that
            touches the share (DESIGN 5.11). An implementation default.
        fatal_pause: How long the process waits after the worker fails before it exits (DESIGN 5.9).
        marker_check_interval: The least time between marker checks during decoding (M18).
        idle_wait: How long the idle worker waits before it checks the marker again. It stays below
            `staging_health_max_age`.
        pause_wait: How long the paused worker waits before it checks the marker again.
    """

    staging_dir: Path
    database: Path
    weights_dir: Path
    profiles: ProfileSet
    host: str = '0.0.0.0'
    port: int = 8000
    job_retention: float = 7 * 24 * 60 * 60
    shutdown_grace: float = 30.0
    staging_health_max_age: float = 120.0
    fatal_pause: float = 60.0
    marker_check_interval: float = 10.0
    idle_wait: float = field(default=30.0)
    pause_wait: float = 5.0

    @classmethod
    def from_environ(cls, environ: Mapping[str, str], platform: str = sys.platform) -> Self:
        """Read the settings from an environment mapping, such as `os.environ`.

        Unset or empty variables take their defaults.

        Raises:
            SettingsError: If a required variable is missing, or a value or the profiles file is invalid.
        """
        profiles_file = environ.get('TRANSCRIBER_PROFILES_FILE') or None
        return cls(
            staging_dir=_path(environ, 'TRANSCRIBER_STAGING_DIR'),
            database=_path(environ, 'TRANSCRIBER_DATABASE'),
            weights_dir=_path(environ, 'TRANSCRIBER_WEIGHTS_DIR'),
            profiles=load_profiles(None if profiles_file is None else Path(profiles_file), platform),
            host=environ.get('TRANSCRIBER_HOST') or '0.0.0.0',
            port=_port(environ.get('TRANSCRIBER_PORT') or '8000'),
            job_retention=_number(environ, 'TRANSCRIBER_JOB_RETENTION_DAYS', 7.0) * 86400,
            shutdown_grace=_number(environ, 'TRANSCRIBER_SHUTDOWN_GRACE', 30.0),
        )


def _path(environ: Mapping[str, str], name: str) -> Path:
    """Return a required path.

    Raises:
        SettingsError: If the variable is unset or empty.
    """
    value = environ.get(name, '')
    if not value:
        raise SettingsError(f'{name} must be set')
    return Path(value)


def _number(environ: Mapping[str, str], name: str, default: float) -> float:
    """Parse a finite, non-negative number, or return the default when unset.

    Raises:
        SettingsError: If the value is not such a number.
    """
    raw = environ.get(name, '')
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise SettingsError(f'{name} must be a number, got {raw!r}') from None
    if not math.isfinite(value) or value < 0:
        raise SettingsError(f'{name} must be a finite non-negative number, got {raw!r}')
    return value


def _port(raw: str) -> int:
    """Parse a TCP port.

    Raises:
        SettingsError: If the value is not an integer from 1 to 65535.
    """
    try:
        port = int(raw)
    except ValueError:
        raise SettingsError(f'TRANSCRIBER_PORT must be an integer, got {raw!r}') from None
    if not 1 <= port <= 65535:
        raise SettingsError(f'TRANSCRIBER_PORT must be from 1 to 65535, got {raw!r}')
    return port
