"""Offline lookup of a profile's model weights in the weights directory (DESIGN section 9, M22).

The weights directory has the layout of a Hugging Face cache, which `hf download --cache-dir`
creates on the host. A profile's model and revision select one snapshot directory. Engine libraries
download whenever they get a name that is not a local directory, so adapters pass them only the
directory that this module resolves. Resolution never touches the network.
"""

from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from media_transcriber.engine import Profile

__all__ = ['WeightsNotFoundError', 'resolve_weights']


class WeightsNotFoundError(FileNotFoundError):
    """The weights directory holds no snapshot for a profile's model and revision."""


def resolve_weights(weights_dir: Path, profile: Profile) -> Path:
    """Return the local snapshot directory of a profile's model at its pinned revision.

    The lookup reads only the weights directory. It neither downloads nor contacts the Hub, and it
    reads no token or other user configuration.

    Args:
        weights_dir: The weights directory, laid out as a Hugging Face cache.
        profile: The profile whose `model` and `revision` select the snapshot.

    Returns:
        The snapshot directory, which holds the model's files.

    Raises:
        WeightsNotFoundError: If the snapshot is not in the weights directory.
        ValueError: If `profile.model` is not a valid repository identifier.
    """
    try:
        snapshot = snapshot_download(
            repo_id=profile.model, revision=profile.revision, cache_dir=weights_dir, local_files_only=True,
        )
    except LocalEntryNotFoundError as error:
        raise WeightsNotFoundError(
            f'No weights for {profile.model} at revision {profile.revision} in {weights_dir}. '
            'Download them on the host before the service loads them (DESIGN section 9).'
        ) from error
    return Path(snapshot)
