# media-transcriber

Transcribe media on a shared staging area into SRT subtitles, as a LAN-only HTTP service on an Apple M-series Mac or an NVIDIA CUDA machine, and benchmark the Whisper engines that do it.

- [DESIGN.md](DESIGN.md) owns the requirements and the design.
- [tasks/TODO.md](tasks/TODO.md) owns the outstanding work.
- [AGENTS.md](AGENTS.md) owns the working standards.

**Status:**

- The governance framework, the devcontainer and the gates are in place ([T1](tasks/closed/T1.md)).
- The engine interface and the transcription pipeline exist ([T3](tasks/closed/T3.md)). No service runs yet. [tasks/TODO.md](tasks/TODO.md) gives the order of the work.
- The CUDA machine is not available yet (DESIGN decision M15).

## Security / trust model

The service will be unauthenticated and intended for a trusted LAN only. Do not expose it to the public internet. [DESIGN section 9](DESIGN.md#9-deployment-and-development-environment) describes the deployment.

## Dev Container (recommended)

Open the repository in VS Code and reopen it in the Dev Container (`.devcontainer/`). [DESIGN section 9](DESIGN.md#9-deployment-and-development-environment) owns the split between the devcontainer and host checks.

**The image.**

- The container's `app` service builds the Dockerfile's `dev` target (`.devcontainer/docker-compose.devcontainer.yaml`).
- That target installs the locked runtime dependencies, the `dev` dependencies and the editable project into the venv at `/opt/venv`. The venv is first on `PATH`.
- If `PYPI_PROXY` is set, every PyPI package is fetched through it.
- The image is Ubuntu 26.04 with its distro Python 3.14. It also contains ffmpeg, uv, git and curl.
- The build fails if `uv.lock` no longer matches `pyproject.toml`.
- The production CUDA image is a later task, [T10](tasks/open/T10.md).

**The creation hook.** The hook installs no Python packages. It does these steps:

1. It checks offline that `uv.lock` still matches `pyproject.toml`, and it fails if they differ. It never rewrites the lock.
2. It fetches GitHub's SSH host keys over HTTPS from `api.github.com/meta` and writes them to `~/.ssh/known_hosts`.
3. It sets up the [shared tools](#shared-tools).
4. It installs Claude Code and standalone Codex with their native installers, and it verifies `codex --version`.

Sign in with `codex login` before you use the direct CLI review. [The review skill](.agents/skills/adversarial-review/SKILL.md#reviewer-session) owns the dispatch.

**Git over SSH.**

- The container mounts your `~/.gitconfig` read-only. It mounts no SSH keys.
- Git over SSH uses the SSH agent that VS Code forwards from the host.
- Load your key into the host's agent, for example with `ssh-add`, before you open the container. Then check with `ssh-add -l` inside it.

**Storage.**

- It needs no `.env` file.
- It bind-mounts the workspace and your `~/.gitconfig`, and it has a `dev_db` volume and a `dev_persist` volume for Claude Code and Codex state.
- It never mounts the staging share, the model weights directory or the video library. Work that needs them is a host check.

## Configuration

Settings arrive with the HTTP API, [T8](tasks/open/T8.md).

## Running the service

The CUDA deployment is [T10](tasks/open/T10.md), and the Mac deployment is [T11](tasks/open/T11.md).

## Updating dependencies

- `pyproject.toml` declares the dependencies: runtime dependencies under `[project]`, and dev tools in the `dev` dependency group.
- `uv.lock` pins the full resolution, including the artifact hashes.
- The lock changes only through deliberate commands: `uv lock`, the update script below, or `uv add`. A rebuild of the container never changes it.

Refresh the lock from the dev container with the update script. The script locks, and then applies the new lock to the container's venv with `uv sync --locked`.

```sh
./update-dependencies.sh                  # move every package to the newest version pyproject.toml allows
./update-dependencies.sh faster-whisper   # move only the named packages
```

- To add a runtime dependency, use `uv add <package>`. To add a dev tool, use `uv add --group dev <package>`. Both update `pyproject.toml`, `uv.lock` and the venv together.
- The script talks to PyPI directly, because a lock resolved through the mirror would record the mirror's URLs. A rebuild of the dev container then installs the new lock through the mirror.

After an update:

1. Review the `uv.lock` diff.
2. Run the gates under [Testing](#testing).
3. Commit `pyproject.toml` and `uv.lock` together.
4. Rebuild the image to deploy the new lock. The image installs only what the lock records.

## Testing

[AGENTS.md](AGENTS.md#validation-and-review) owns when the gates are required. [tests/README.md](tests/README.md) owns how tests are written. The gates are:

```sh
./check-coverage.sh
mypy media_transcriber
ruff check .
```

**The coverage gate.**

- `check-coverage.sh` runs pytest. It requires at least 80% statement coverage and branch coverage, for every measured file and for the project.
- Run it from the repository root.
- It refuses to run unless `media_transcriber` resolves to this tree's package.
- It also refuses to run unless `python3` matches the version that `.python-version` pins. A pin of `3.14` accepts any 3.14.x.
- Without that check, a dev container built before a Python upgrade would pass the suite on its old interpreter. Rebuild the container when the check fails.
- `COVERAGE_THRESHOLD` overrides the script's threshold with a number from 0 to 100 inclusive. An empty value keeps 80.
- A value that is not numeric, not finite or out of range fails before pytest runs.
- pytest also keeps its configured aggregate floor of 80%.
- Pass only whole-suite options to this gate, such as `-x` or `-q`. Selected test paths cannot certify per-file coverage.
- Coverage includes the application package and the local mutation and worktree gate helpers. The shared repository measures coverage of its own tools.

**During development.**

- Tests run in random order through `pytest-randomly`. Reproduce an ordering failure with the printed `--randomly-seed=<seed>`.
- Use focused checks: `pytest --no-cov <paths>`. The option avoids the aggregate coverage failure that a partial selection causes.
- Inspect the per-file coverage and the warnings in the final run.

**A separate worktree.**

- Run the gates in the tree that is validated.
- In a separate git worktree, set `PYTHONPATH` to the worktree's root. A worktree that holds a frozen candidate for review is an example.
- Without that setting, the dev container's editable install supplies the main tree's package. `check-coverage.sh` then refuses to run, because it would measure the wrong files.

**Checks that need Docker.**

- The dev container has no Docker daemon. Image builds and container creation are therefore host checks, like the other host checks that [DESIGN section 9](DESIGN.md#9-deployment-and-development-environment) defines.
- Record host checks in the host-check task, [T2](tasks/open/T2.md).
- `docker compose config` needs no daemon. To validate Compose changes in the dev container, do these steps:
  1. Download the standalone Compose binary for the session from the Compose GitHub release.
  2. Check the binary against the SHA-256 that the release publishes.
  3. Run it in a scratch copy of the repository without `.env`, with `COMPOSE_PROJECT_NAME` unset.

**The documentation check.** [AGENTS.md](AGENTS.md#validation-and-review) requires it for every change that touches a Markdown file.

- Run `../agent-workflow-common/tools/check_docs.py` from the repository. It takes less than a second.
- It checks that each relative link and each anchor resolves.
- It checks the task files against the index in `tasks/TODO.md`.
- It checks the rules of [STYLE.md](STYLE.md) that need no judgment, such as the sentence limits.
- In `DESIGN.md`, `project-reviews.md` and the task files, it checks the style of changed lines only.
- It prints each problem as `path:line: rule: message`.
- A pass means that the local documentation references resolve. It does not mean that the documents agree with each other.

**The reference report.** It is optional. Use it before you rename or move a file or a heading.

- Run `../agent-workflow-common/tools/check_docs.py --references AGENTS.md` to list each line that links to that file or to one of its headings.
- Run `../agent-workflow-common/tools/check_docs.py --references 'AGENTS.md#governance-files'` to list only the links to that heading.
- Write the path from the root of the repository.
- The report runs no check. It lists Markdown links only, not paths in code or in plain text.

**Tool configuration.** `pyproject.toml` configures the tools.

- pytest rejects unregistered markers and unexpected passes of `xfail` tests.
- pytest turns every warning into a test failure. Assert an expected warning with `pytest.warns`.
- Ruff enforces pycodestyle (120-character lines), Pyflakes, import order, bugbear, pyupgrade and Google-style docstrings.
- mypy reports missing imports as errors.
- A dependency that ships no type information gets a narrow `ignore_missing_imports` override when the application first imports it.
- Tests are intentionally outside the mypy gate.

## Shared tools

Three tools run from the sibling checkout `../agent-workflow-common`. This repository holds their configuration in `agent-workflow.json` and keeps no copies of their implementations or tests.

| Tool | Purpose |
|---|---|
| `../agent-workflow-common/tools/check_docs.py` | Documentation checks and reference reports |
| `../agent-workflow-common/tools/snapshot.py` | Review integrity snapshots |
| `../agent-workflow-common/tools/dispatch.py` | Codex CLI reviewer dispatch |

- The devcontainer runs `.devcontainer/postCreateCommand.sh`, which clones the tools into `$HOME/persist/agent-workflow-common` on the existing `dev_persist` volume.
- The hook changes only `/workspaces` ownership to `ubuntu:ubuntu` with `sudo chown`, then links the checkout at `/workspaces/agent-workflow-common`.
- First setup needs GitHub access. Existing checkouts are reused without pulls or resets, so local work survives recreation of the container.
- Setup stops if the sibling path conflicts, the checkout is invalid, required tools are missing, or cloning fails. It preserves existing paths.
- Resolve a reported conflict explicitly. The hook never replaces another checkout or removes an incomplete one.
- This checkout is separate from the checkouts of `youtube-collector` and ClipboardTTS. Commit and push shared changes to transfer them between environments.
- Temporary review evidence stays outside the repository. Governance, documentation and durable review records remain tracked but are excluded from Docker's build context.
- The tools use Python 3.9 or later and the standard library. Their `--help` text describes the arguments.
- The checkout determines the tool version. No version is pinned, matching `youtube-collector` and ClipboardTTS.
- Change shared tools and run their tests in their repository. Verify every consumer after shared changes.
- Keep `mutate.py` and `worktree_gates.py` local. They implement this project's Python test isolation and gates.
- From a review worktree, use the shared checkout's absolute tool path and run it with the worktree as the current directory.
- For CLI dispatch from another directory, pass the worktree with `--target`.
- [project-reviews.md](project-reviews.md) preserves historical project reviews. Task state remains in [tasks/](tasks/).
