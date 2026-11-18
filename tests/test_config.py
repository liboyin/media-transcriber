"""Tests for the settings and the profiles file (DESIGN 5.3, section 9, M24)."""

from pathlib import Path

import pytest

import media_transcriber.config as testee
from media_transcriber import faster_whisper_adapter, mlx_whisper_adapter
from media_transcriber.engine import Profile

REQUIRED = {
    'TRANSCRIBER_STAGING_DIR': '/staging', 'TRANSCRIBER_DATABASE': '/db/t.sqlite3', 'TRANSCRIBER_WEIGHTS_DIR': '/w',
}

TWO_PROFILES = '''
default = "fast"

[[profiles]]
name = "slow"
engine = "faster-whisper"
model = "org/slow"
revision = "r1"
precision = "float32"

[[profiles]]
name = "fast"
engine = "mlx-whisper"
model = "org/fast"
revision = "r2"
precision = "float16"
decoding = { beam_size = 5, temperature = 0.0, prompt = "x", fp16 = true }
'''


def write(tmp_path: Path, text: str) -> Path:
    """Write a profiles file into test-owned storage."""
    path = tmp_path / 'profiles.toml'
    path.write_text(text)
    return path


def test_from_environ_reads_required_paths_and_defaults() -> None:
    """With only the required variables, every other setting takes its default and the baseline is offered."""
    settings = testee.Settings.from_environ(REQUIRED, platform='linux')
    assert (settings.staging_dir, settings.database, settings.weights_dir) == (
        Path('/staging'), Path('/db/t.sqlite3'), Path('/w'))
    assert (settings.host, settings.port) == ('0.0.0.0', 8000)
    assert settings.job_retention == 7 * 86400
    assert settings.shutdown_grace == 30.0
    baseline = faster_whisper_adapter.BASELINE_PROFILE
    assert settings.profiles == testee.ProfileSet({baseline.name: baseline}, baseline.name)
    # The idle worker must check the marker again before health would call its check stale.
    assert settings.idle_wait < settings.staging_health_max_age


def test_from_environ_reads_every_optional_variable(tmp_path: Path) -> None:
    """Each optional variable overrides its default."""
    environ = {
        **REQUIRED, 'TRANSCRIBER_PROFILES_FILE': str(write(tmp_path, TWO_PROFILES)),
        'TRANSCRIBER_HOST': '127.0.0.1', 'TRANSCRIBER_PORT': '9000', 'TRANSCRIBER_JOB_RETENTION_DAYS': '0.5',
        'TRANSCRIBER_SHUTDOWN_GRACE': '2.5',
    }
    settings = testee.Settings.from_environ(environ)
    assert (settings.host, settings.port, settings.job_retention, settings.shutdown_grace) == (
        '127.0.0.1', 9000, 43200.0, 2.5)
    assert list(settings.profiles.profiles) == ['slow', 'fast']


@pytest.mark.parametrize('name', sorted(REQUIRED))
@pytest.mark.parametrize('value', [None, ''])
def test_from_environ_requires_the_paths(name: str, value: str | None) -> None:
    """A missing or empty required variable fails at startup, naming it."""
    environ = {key: item for key, item in REQUIRED.items() if key != name}
    if value is not None:
        environ[name] = value
    with pytest.raises(testee.SettingsError, match=name):
        testee.Settings.from_environ(environ)


@pytest.mark.parametrize(('name', 'value'), [
    ('TRANSCRIBER_PORT', 'eighty'), ('TRANSCRIBER_PORT', '0'), ('TRANSCRIBER_PORT', '65536'),
    ('TRANSCRIBER_JOB_RETENTION_DAYS', 'week'), ('TRANSCRIBER_JOB_RETENTION_DAYS', '-1'),
    ('TRANSCRIBER_SHUTDOWN_GRACE', 'inf'), ('TRANSCRIBER_SHUTDOWN_GRACE', 'nan'),
])
def test_from_environ_rejects_invalid_values(name: str, value: str) -> None:
    """A malformed or out-of-range value fails at startup, naming its variable."""
    with pytest.raises(testee.SettingsError, match=name):
        testee.Settings.from_environ({**REQUIRED, name: value})


@pytest.mark.parametrize(('platform', 'expected'), [
    ('darwin', mlx_whisper_adapter.BASELINE_PROFILE), ('linux', faster_whisper_adapter.BASELINE_PROFILE),
])
def test_without_a_file_the_platform_baseline_is_the_only_profile(platform: str, expected: Profile) -> None:
    """The MLX baseline is offered on macOS, and the faster-whisper baseline elsewhere (M24)."""
    assert testee.load_profiles(None, platform) == testee.ProfileSet({expected.name: expected}, expected.name)


def test_load_profiles_reads_every_field_in_order(tmp_path: Path) -> None:
    """Profiles keep the file's order, their decoding settings and the named default."""
    profiles = testee.load_profiles(write(tmp_path, TWO_PROFILES), 'linux')
    assert profiles.default == 'fast'
    assert list(profiles.profiles.values()) == [
        Profile('slow', 'faster-whisper', 'org/slow', 'r1', 'float32'),
        Profile('fast', 'mlx-whisper', 'org/fast', 'r2', 'float16',
                {'beam_size': 5, 'temperature': 0.0, 'prompt': 'x', 'fp16': True}),
    ]


def test_load_profiles_defaults_to_the_first_profile(tmp_path: Path) -> None:
    """Without `default`, the first profile is the default."""
    profiles = testee.load_profiles(write(tmp_path, TWO_PROFILES.replace('default = "fast"', '')), 'linux')
    assert profiles.default == 'slow'


PROFILE = '[[profiles]]\nname = "a"\nengine = "faster-whisper"\nmodel = "m"\nrevision = "r"\nprecision = "float32"\n'


@pytest.mark.parametrize(('text', 'message'), [
    ('this is not toml', 'cannot be read'),
    ('', 'at least one'),
    ('profiles = []', 'at least one'),
    ('profiles = "a"', 'at least one'),
    ('profiles = [1]', 'is not a table'),
    ('extra = 1\n' + PROFILE, 'unknown keys'),
    (PROFILE + 'colour = "red"\n', 'unknown keys'),
    (PROFILE.replace('model = "m"\n', ''), '`model`'),
    (PROFILE.replace('"float32"', '""'), '`precision`'),
    (PROFILE.replace('"float32"', '32'), '`precision`'),
    (PROFILE.replace('faster-whisper', 'whisper.cpp'), 'unknown engine'),
    (PROFILE + 'decoding = "greedy"\n', '`decoding`'),
    (PROFILE + 'decoding = { beams = [1, 2] }\n', '`decoding`'),
    (PROFILE + PROFILE, 'twice'),
    ('default = "b"\n' + PROFILE, "default 'b'"),
    ('default = ["a"]\n' + PROFILE, "default \\['a'\\]"),
])
def test_load_profiles_rejects_a_malformed_file(tmp_path: Path, text: str, message: str) -> None:
    """Every malformed file fails at startup with a message that names the problem and the file."""
    path = write(tmp_path, text)
    with pytest.raises(testee.SettingsError, match=message) as raised:
        testee.load_profiles(path, 'linux')
    assert str(path) in str(raised.value)


def test_load_profiles_reports_a_missing_file(tmp_path: Path) -> None:
    """A profiles file that does not exist fails at startup."""
    with pytest.raises(testee.SettingsError, match='cannot be read'):
        testee.load_profiles(tmp_path / 'absent.toml', 'linux')
