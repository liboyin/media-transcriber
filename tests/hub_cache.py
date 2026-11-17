"""A test-owned weights directory in the layout of a Hugging Face cache (DESIGN M22).

`hf download --cache-dir` stores a repository's files under `models--<org>--<name>/snapshots/<revision>/`.
An offline lookup by commit hash needs only that snapshot directory, so tests build it without the Hub.
"""

from pathlib import Path


def add_snapshot(weights_dir: Path, model: str, revision: str) -> Path:
    """Create the snapshot directory of a model at a revision, with a placeholder config, and return it."""
    snapshot = weights_dir / f'models--{model.replace("/", "--")}' / 'snapshots' / revision
    snapshot.mkdir(parents=True)
    (snapshot / 'config.json').write_text('{}')
    return snapshot
