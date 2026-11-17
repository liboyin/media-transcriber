# Stages: `base` holds what the application needs at run time; `build` adds git and uv and builds
# the locked venv; and `dev` continues from `build` with the locked dev group and an editable
# project for the dev container (selected in .devcontainer/docker-compose.devcontainer.yaml).
# The CUDA runtime stage for the production service is a later task, because the CUDA machine
# is not available yet (DESIGN decision M15). Every PyPI package, dev tools included, is
# installed here through PYPI_PROXY when it is set, so creating the dev container needs no PyPI
# access. This follows youtube-collector's build design. Ubuntu 26.04's distro python3 is
# Python 3.14 (DESIGN assumption A3). Base images are pinned by tag for readability and by
# digest so the pin is immutable.
FROM ubuntu:26.04@sha256:6a06111c3b0a8600feadb270aec771dc239bf407b25625f93f44ab00e11f4d3c AS base

ARG APT_PROXY

# The application runs from a uv-managed venv built from the distro's python3; uv never
# downloads an interpreter. The venv lives outside the project directory because the dev
# container bind-mounts the workspace over it, which would hide an in-project .venv.
# UV_PROJECT_ENVIRONMENT points the dev container's `uv sync` (update-dependencies.sh) at the
# same venv.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# The apt/uv steps use the shared build helpers, supplied as the named build context
# `build_common` and pinned in .devcontainer/docker-compose.devcontainer.yaml. They are bind-mounted rather than copied:
# they are build-time only, so nothing is left behind in the image. A build that bypasses
# compose must supply the context itself, otherwise BuildKit looks for an image by that name
# on Docker Hub:
#   docker buildx build --target dev --build-context build_common=<git-url-or-local-clone> .
#
# ffmpeg decodes media into the waveform that the engines transcribe. Plain Ubuntu has no CA
# certificates, and the helper skips recommended packages, so ca-certificates is listed
# explicitly: without it every HTTPS request, at build and at run time, fails verification.
RUN --mount=type=bind,from=build_common,target=/build-common \
    /build-common/apt_install.sh ca-certificates ffmpeg python3

# The venv and the project directory belong to the runtime user from the start, so the dev
# container's `uv sync` (update-dependencies.sh) can update the venv without a recursive
# chown, which would copy the whole venv into another layer. The database directory is the
# mount point of a named volume; Docker initializes a new volume from the image's directory,
# ownership included, so the runtime user can write there. No staging directory exists in the
# image: the dev container mounts no share (DESIGN section 9).
RUN install -d -o ubuntu -g ubuntu /opt/venv /workspaces/media-transcriber \
    /var/lib/media-transcriber
WORKDIR /workspaces/media-transcriber

FROM base AS build

ARG APT_PROXY
ARG PYPI_PROXY

# git fetches git dependencies, such as pathlib-extensions once the staging task adds it, and
# the review tools and their tests run it in the dev container. It and uv are build and
# development tools, so the future runtime stage leaves them out.
RUN --mount=type=bind,from=build_common,target=/build-common \
    /build-common/apt_install.sh git

# uv stays in `dev` because the dev container runs it (`uv sync`, `uv lock`). The tag names
# the version; the digest makes the pin immutable.
COPY --from=ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 /uv /usr/local/bin/uv

USER ubuntu

# Install pinned Python dependencies before copying the source, so editing source does not
# invalidate this layer and force a full dependency reinstall on every rebuild. The helper
# exports the lock with --frozen, which never checks it against pyproject.toml, so check that
# first: a stale lock fails the build instead of installing outdated pins. --offline keeps the
# check off the network; an up-to-date lock is confirmed from the files alone, and a stale one
# fails.
COPY --chown=ubuntu pyproject.toml uv.lock .python-version ./
RUN UV_NO_CACHE=1 uv lock --check --offline
RUN --mount=type=bind,from=build_common,target=/build-common \
    /build-common/uv_install.sh /opt/venv --no-emit-project

# Copy the rest of the source and install the project itself as a regular package, so that
# the future runtime stage can copy this venv without the source tree or uv. `dev` replaces it
# with an editable install.
COPY --chown=ubuntu . .
RUN --mount=type=bind,from=build_common,target=/build-common \
    /build-common/uv_install.sh /opt/venv --no-editable

FROM build AS dev

ARG APT_PROXY
ARG PYPI_PROXY

# curl runs the native Claude Code and Codex installers and fetches GitHub's SSH host keys in
# .devcontainer/postCreateCommand.sh.
USER root
RUN --mount=type=bind,from=build_common,target=/build-common \
    /build-common/apt_install.sh curl
USER ubuntu

# Mount point of the dev container's dev_persist volume (Claude Code and Codex state),
# created by the runtime user so a new volume initializes writable for it.
RUN install -d /home/ubuntu/persist

# uv pip sync makes the venv match the export exactly, so this keeps the runtime packages,
# adds the dev group, and replaces the regular project install with an editable one, which
# the dev container's workspace bind mount needs.
RUN --mount=type=bind,from=build_common,target=/build-common \
    /build-common/uv_install.sh /opt/venv --group dev
