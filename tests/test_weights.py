"""Tests for the offline lookup of model weights in a test-owned weights directory."""

from pathlib import Path

import pytest
from hub_cache import add_snapshot
from huggingface_hub.errors import LocalEntryNotFoundError

import media_transcriber.weights as testee
from media_transcriber.engine import Profile

REVISION = '0123456789abcdef0123456789abcdef01234567'
PROFILE = Profile(name='p', engine='e', model='org/model', revision=REVISION, precision='float32')


def test_resolve_weights_returns_the_snapshot_of_the_pinned_revision(tmp_path: Path) -> None:
    """The profile's model and revision select one snapshot directory in the cache layout."""
    snapshot = add_snapshot(tmp_path, 'org/model', REVISION)
    add_snapshot(tmp_path, 'org/model', 'f' * 40)

    assert testee.resolve_weights(tmp_path, PROFILE) == snapshot


def test_resolve_weights_refuses_another_revision_of_the_model(tmp_path: Path) -> None:
    """Weights of the same model at another revision do not satisfy the pin."""
    add_snapshot(tmp_path, 'org/model', 'f' * 40)

    with pytest.raises(testee.WeightsNotFoundError) as caught:
        testee.resolve_weights(tmp_path, PROFILE)

    assert isinstance(caught.value, FileNotFoundError)
    assert isinstance(caught.value.__cause__, LocalEntryNotFoundError)
    assert 'org/model' in str(caught.value)
    assert REVISION in str(caught.value)
    assert str(tmp_path) in str(caught.value)


def test_resolve_weights_never_contacts_the_hub(tmp_path: Path, network_refusals) -> None:
    """Neither a found nor a missing snapshot makes a network attempt (DESIGN section 9).

    The Hub library falls back to its cache when a lookup fails, so a refused attempt does not change
    the result. The guard's record is therefore the only evidence, and the test asserts it.
    """
    add_snapshot(tmp_path, 'org/model', REVISION)
    try:
        testee.resolve_weights(tmp_path, PROFILE)
        with pytest.raises(testee.WeightsNotFoundError):
            testee.resolve_weights(tmp_path, Profile('p', 'e', 'org/model', 'f' * 40, 'float32'))
    finally:  # taken even after a failure, so that the failure stays the only report
        refusals = network_refusals.take()
    assert refusals == []


def test_resolve_weights_creates_nothing_when_the_directory_is_missing(tmp_path: Path) -> None:
    """A missing weights directory fails the lookup and is not created, so the lookup only reads."""
    weights_dir = tmp_path / 'weights'

    with pytest.raises(testee.WeightsNotFoundError):
        testee.resolve_weights(weights_dir, PROFILE)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('model', ['../model', 'org/model/extra', ''])
def test_resolve_weights_rejects_an_invalid_repository(tmp_path: Path, model: str) -> None:
    """A model that is not a repository identifier fails before any path is built from it."""
    profile = Profile(name='p', engine='e', model=model, revision=REVISION, precision='float32')

    with pytest.raises(ValueError, match='Repo id'):
        testee.resolve_weights(tmp_path, profile)
