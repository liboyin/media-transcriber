# media-transcriber: Requirements and Design

This document owns the product requirements, system design, product contracts and accepted user decisions. [README](README.md) will cover setup and operation. Task files, AGENTS and the repository skills will own outstanding work and working procedures once the governance scaffolding exists (section 11).

Status: accepted requirements, written 2026-11-17 in a design discussion that started from `TRANSCRIPTION_REDESIGN.md` in `youtube-whisperer` at revision `474f5dc` (the "source revision"). Section 16 records what is implemented. Section 13 records the user's decisions. Section 15 records assumptions that stand unless the user objects. Record later decisions with new IDs after the last one in section 13, and record the architecture in a new section as components are built.

Requirement words (MUST, SHOULD, MAY) follow BCP 14. Behavior inherited from `youtube-whisperer` cites the source revision. It describes what the code did there, which this project reimplements rather than imports.

## 1. Purpose and context

`youtube-whisperer` currently downloads YouTube media, transcribes media that lacks captions, and stores everything in one staging directory. The user is splitting it up. `youtube-collector` has taken over YouTube acquisition. `media-transcriber` takes over transcription. `youtube-whisperer` becomes an orchestrator that calls both.

The project serves two purposes (M1):

1. **Transcriber service.** A LAN-only HTTP service that turns a media file on the shared staging area into an SRT beside it. It runs either natively on an Apple M-series Mac or in Docker on an NVIDIA CUDA machine.
2. **Benchmark tool.** A command that measures how long the same Whisper model takes on the same audio. It uses the service's engine code and measures time only (M12).

The redesign handoff left the backend, the interface, the queue and file access open. This document settles all four. The initial engines are MLX Whisper on Apple silicon and faster-whisper on CUDA (M13). Tuning each platform's default profile follows benchmark results (M5).

## 2. Goals and non-goals

Goals:

- Transcribe one media file into one SRT, prioritizing the time to finish a single file over throughput across many files (D53 at the source revision).
- Run on Apple silicon with native GPU acceleration, and on NVIDIA CUDA, from one codebase (M3).
- Give each submission a job identifier with its state, outcome and failure reason, followed by polling.
- Publish the SRT beside its media on the staging share, atomically, without ever overwriting a file (M7, M9).
- Measure the speed of engine profiles reproducibly, so that a later run can be compared with an earlier one on the same machine (M5).

Non-goals for this version:

- Re-transcribing over an existing SRT, or recording which engine produced an SRT (M9).
- Azure Speech or any other cloud engine. Azure is a planned engine, postponed (M6).
- Automatic fallback from one engine or machine to another. A caller names a profile or takes the deployment's default.
- Language auto-detection. The caller states the source language (M8).
- Translation, diarization, word-level timestamps, and output formats other than SRT.
- Choosing between the Mac and the CUDA machine. The orchestrator routes jobs (M8).
- Deciding whether a video needs transcription. The orchestrator decides from the collector's caption outcome.
- Cancellation, automatic retry of a job that has failed, authentication, multi-user use, exposure beyond a trusted LAN, or horizontal scaling. Restart recovery deliberately repeats an interrupted job (5.9).
- Backward compatibility with `youtube-whisperer`'s Python entry points (assumption A1).
- Claiming that results from different machines measure the same engine on different hardware (M5).
- Measuring transcript quality, such as word error rate, in the benchmark (M12).

## 3. Consumers

| Consumer | Uses the project to |
|---|---|
| `youtube-whisperer` orchestrator | Submit one media file per job to the Mac or the CUDA service, poll the job, and record failures on its own side |
| The user | Submit jobs directly, check failures, clear reviewed failures, and run benchmarks to choose and re-check engine profiles |

Orchestrator workflow:

1. The collector publishes `<videoID> - <Title>.mp4` into staging and reports captions `unavailable` or `error`.
2. The orchestrator chooses a deployment and submits the MP4's path relative to the staging root, with the source language.
3. The transcriber returns a job identifier. The orchestrator reads the job at regular intervals.
4. When the job's turn arrives, the transcriber publishes `<videoID> - <Title>.srt` beside the MP4.
5. The next read shows the job `done` with the outcome `transcribed`, or `existing` when an SRT was already there.
6. The user moves the pair into the permanent video library.

Benchmark workflow:

1. The user picks recordings from the video library: one English and one Chinese (7.2).
2. On one machine, the user runs the benchmark with a list of engine profiles.
3. The tool writes a result set (7.5). The user compares it with earlier result sets, from the same machine or from another one.

## 4. Terms

- **Staging area:** the shared directory that the collector writes into, an SMB share in the user's deployment. Each host mounts it at its own path, configured as the staging root.
- **Media file:** any file in the staging area with an audio track that ffmpeg can decode.
- **Engine:** an adapter around one inference library, such as faster-whisper or MLX Whisper.
- **Engine profile:** a named, fixed combination of engine, model and revision, precision, and decoding settings. Profiles are the unit that the service offers and that the benchmark measures.
- **Deployment:** one running service process on one host, with its configured profiles and default profile.
- **Job:** one accepted submission: a media path, a source language and a profile.
- **Failure:** a self-contained record that a job did not succeed, kept until the user clears it.
- **Corpus:** the set of recordings that the benchmark runs.
- **Result set:** the files that one benchmark run writes.

## 5. Service functional requirements

### 5.1 Submission

`POST /jobs` accepts:

- `path` (required): the media file's path relative to the staging root.
- `language` (required): the source language as a BCP 47 tag (5.4).
- `profile` (optional): the name of one of the deployment's profiles. If it is absent, the deployment's default profile applies.

A valid submission returns 202 with the job's initial representation and identifier. The API validates syntactically only and MUST NOT touch the staging share, as in the collector's C25: a stalled SMB share must not stall the API. A path that is absolute, contains `..` or a hidden component, or is empty returns 422. An unknown profile, or a language that the profile does not support, returns 422. Every valid submission creates a new job. Submitting the same file again is harmless, because the later job finds the published SRT `existing`.

### 5.2 Paths and the staging root

- The staging root is configured per deployment. The CUDA deployment bind-mounts the share into its container. The Mac deployment reads the share through its macOS SMB mount, such as a path under `/Volumes` (M7).
- Paths in requests, responses and failure records are relative to the staging root, so the collector, the orchestrator and each transcriber may mount the share at different paths.
- The worker MUST resolve a path inside the staging root and MUST reject a path that resolves outside it, including through a symlink.

### 5.3 Engine profiles

- Each deployment configures the profiles it offers and its default profile (M8). The configuration names each profile's engine, model and revision, precision and decoding settings.
- `GET /profiles` lists the deployment's profiles, their settings and their supported languages, without touching the share or loading a model.
- The service holds at most one resident model at a time. A job for another profile replaces the resident model (M8).
- Initial engines are listed in section 8. Choosing each platform's default profile is a benchmark decision (M5).

### 5.4 Language

- The caller states the source language. The service does not detect it (M8).
- Any well-formed BCP 47 tag is accepted if the profile's engine supports its language. There is no fixed whitelist like the source revision's four tags.
- The engine adapter maps the tag to its own code. For Whisper engines that is the primary language subtag, so `zh-CN`, `zh-TW` and `zh` all become `zh`.
- The job records both the requested tag and the code the engine received.

### 5.5 Processing a job

The worker processes one job at a time (M8). For each job it:

1. Checks the staging marker (5.10).
2. Resolves the media path and the SRT path. The SRT path is the media path with the suffix `.srt`.
3. Applies the `NEVER` policy: if a file already exists at the SRT path, the job ends `done` with the outcome `existing`, before any decoding or model work (M9).
4. Decodes the audio with ffmpeg into a mono 16 kHz float32 waveform, as at the source revision (`waveform_loader.py`). An empty waveform fails the job before any model work.
5. Loads the profile's model if it is not resident.
6. Transcribes the waveform, consuming every lazy segment before staging, as at the source revision.
7. Applies the rejection policy: if any segment contains a known bad phrase, the whole run is rejected and no SRT is written. The policy comes from `rejection_policy.py` and applies to every engine.
8. Serializes the segments to SRT and publishes the file (5.6).

The engine adapter returns timed segments in an engine-neutral form. SRT is an export of those segments, not the engine's output format.

### 5.6 Output and publication

- The SRT is written beside its media, with the media's stem and the suffix `.srt`. Only an exact name match counts as an existing SRT.
- Publication MUST be atomic, as at the source revision and in the collector. The complete text is written in a fresh hidden `.staging-<random>` directory beside the destination and published with pathlib-extensions' `StagedFile` under `NEVER`. A competing file that appears during the job is preserved, and the job ends `existing`. On a filesystem without hard links, such as the Mac's SMB mount, a competing file can be replaced instead. That happens only if it appears between the final check and the rename (M21).
- If the media file is gone when the SRT is about to be published, the job fails with category `missing` and nothing is written (M20).
- A failure or interruption leaves previous output intact and removes its own stage directory. A stage directory left by an abrupt exit is not scavenged.
- SRT formatting follows the source revision's shared formatter: millisecond rounding before decomposition, hours always present, comma milliseconds, and outer whitespace stripped from text. Consecutive segments with identical text become one cue (M19).
- The transcriber writes only new SRT files. It never deletes, renames or replaces anything in staging.

### 5.7 Job states and outcomes

```text
Job: queued → running → done | failed
```

| Transition | When |
|---|---|
| created as `queued` | The submission is accepted |
| `queued` → `running` | The worker starts the job |
| `running` → `done` | The SRT is published (`transcribed`) or already exists (`existing`) |
| `running` → `failed` | Any step in 5.5 fails, which creates a failure record |
| `running` → `queued` | The service restarts while the job runs (5.9) |

A `done` job records its outcome, the profile used, the language code the engine received, the SRT's relative path, and the job's durations: queue wait, model load, decoding, transcription and publication. A job's representation embeds its uncleared failure.

### 5.8 Failures, following a job and retention

These follow the collector's sections 5.8 to 5.10 (assumption A2):

- A caller follows a job by reading `GET /jobs/{id}` at its own interval. There is no long polling and no push channel.
- A job stays readable while it is `queued` or `running`, and for a retention period after it ends, assumed 7 days. The submission transaction deletes expired jobs.
- A failure records the job, the path, the profile, the phase (`staging`, `model`, `decoding`, `transcription`, `rejection` or `publication`), a category and the raw message.
- Categories are `missing` (the media file is gone), `decoding`, `rejected`, `model` (loading failed or the device is unavailable), `filesystem` and `internal`.
- A failure stays until the user clears it, or until a later job for the same path ends `done`.

### 5.9 Restart recovery

- On startup, a job left `running` returns to `queued`. Because of the `NEVER` check, repeating a job that had already published its SRT is harmless.
- On SIGTERM, the worker stops taking jobs. Inference that cannot be interrupted is abandoned when the shutdown grace ends, and recovery requeues it on the next start.
- An error that escapes the worker's handling of a job ends the process with a failure status, after a fixed pause, as in the collector's C53. Recovery counts interruptions and fails a job at the third, as in the collector's C59.
- On the Mac, a supervisor restarts the process: launchd or an equivalent the builder chooses (assumption A4). In Docker, `restart: unless-stopped` does.
- A Mac that sleeps or shuts down only delays queued jobs. The SQLite queue persists them.

### 5.10 Staging availability

- The staging root MUST contain the marker file `.media-transcriber-root`, which the user creates once (M8). The transcriber never creates it.
- Before starting each job and before publishing, the worker MUST check that the marker exists.
- While the marker is missing or the root cannot be read, the worker pauses without changing any record and resumes on its own. On macOS an unmounted SMB share leaves an empty directory or no directory under `/Volumes`, so this guard is required on both platforms.
- A job already running when the share disappears fails with category `filesystem`.

### 5.11 Health

`GET /health` MUST report, without touching the share:

- whether the database is readable and writable
- the worker's most recent marker check and its age, and therefore whether work is paused
- whether the worker thread runs and has not failed
- the worker's current activity, the job it serves, when the activity started, and when the worker last reported progress
- the platform, the configured profiles, the resident model if any, and the engine library versions
- whether the accelerator that each profile needs is available, such as a CUDA device or Metal

A running thread is not proof of progress, because a thread blocked on a stalled share still runs. Health therefore follows the collector's C55 rule, not its older C43 rule (decision M17):

- The worker records whether its current activity touches the share. Idle waiting, the marker check, resolving paths, decoding and publication touch it. Loading the model and transcribing do not, because the waveform is already in memory.
- While the worker is idle or touches the share, `/health` compares the age of its last marker check with a staleness limit (M18).
- During decoding and publication, the worker checks the marker again at an interval, between chunks of ffmpeg's output or of the written file. Progress alone does not refresh the check, because a successful read or write does not show that the marker is still present.
- During an activity that does not touch the share, `/health` reports the last check, its age and the activity's duration for information only, without calling them stale. A long transcription is normal and is not unhealthy.
- The service cannot tell a slow transcription from a hung one. `/health` reports the duration and leaves the judgment to the user.

It returns 200 when everything is healthy and 503 otherwise, with the same body in both cases. The staleness limit is an implementation default.

## 6. HTTP API summary

| Endpoint | Purpose |
|---|---|
| `POST /jobs` | Submit a transcription job (5.1) |
| `GET /jobs` | Every retained job with its uncleared failure |
| `GET /jobs/{id}` | One job (5.7, 5.8) |
| `GET /profiles` | The deployment's engine profiles (5.3) |
| `GET /failures` | All uncleared failures, oldest first |
| `DELETE /failures` | Clear all failures |
| `DELETE /failures/{id}` | Clear one failure |
| `GET /health` | Service health and paused state (5.11) |

An unknown job or failure returns 404. Validation errors return 422. Exact JSON schemas are left to implementation. FastAPI's OpenAPI document at `/docs` publishes them.

## 7. Benchmark

### 7.1 Purpose and uses

The benchmark is a re-runnable regression check (M5). It answers: "did a dependency, engine, model conversion or OS update make transcription slower on this machine?" Two other uses build on it:

- **Backend selection:** one run with candidate profiles alongside the current default. The user records the chosen default profile per platform as a decision in this file.
- **Machine comparison:** result sets from two machines, read side by side. Different platforms run different engines, so such a comparison is of deployments, not of hardware alone.

### 7.2 Corpus

- The repository contains no benchmark recordings (M14). The user keeps one English and one Chinese recording in the video library, and names them and their languages on the command line.
- The benchmark only reads a recording. It MUST NOT write, move or rename anything in the video library.
- Each result set records every recording's SHA-256 and duration. The comparison command warns when two result sets used different recordings.
- The benchmark MUST NOT use the staging share or the service's database as its input or its output.

### 7.3 What one run does

1. Record the environment: the chip or GPU model and core count, installed memory, OS version and drivers. Also record the power source where readable, the Python and engine versions, and the model's identity and hash.
2. For each profile, measure a cold start: process start, model load and first transcription.
3. For each recording and profile, run warm repetitions with the model resident, after a warm-up. The number of repetitions is a setting.
4. Run the same pipeline function the service worker runs: 5.5 steps 4 and 6 to 8, with the model already loaded. The measured time covers decoding, transcription, rejection and SRT serialization. Write the SRT to the result set, not to staging.
5. Consume every lazy output and synchronize asynchronous device work before stopping a timer.
6. Keep every raw timing. Report the median and the spread.

### 7.4 Metrics

The headline metric is wall time from the start of decoding to a complete SRT, for one recording (M11). The benchmark also reports, as diagnostics:

- decoding, inference and serialization times separately
- the real-time factor (processing time divided by audio duration) and its inverse
- cold model-load time
- failures and rejections

The benchmark measures time only (M12). It computes no quality score and needs no reference transcripts. A profile can therefore look faster because it transcribes worse, for example by dropping speech at chunk boundaries. The user judges quality by reading the produced SRTs.

The benchmark measures neither energy nor memory use, and the project records no threshold for "materially faster" (M14). The user reads the timings and decides.

### 7.5 Result sets

- One run writes one result set in machine-readable files. It holds the environment record, the recordings' hashes and durations, the profiles with their settings, the raw timings and the produced SRTs.
- A comparison command reads two or more result sets and reports differences per profile and recording.
- Result sets are written outside the repository, to a directory given on the command line. Whether any part is committed is open (Q4).

### 7.6 Execution

- The benchmark is a subcommand of the project's command-line entry point, beside the command that starts the service.
- It runs only on a host with the real accelerator. Each run is a host check that needs the user's authorization (M10, section 9).
- It never runs inside the automated test suite. Tests cover its logic with fake engines and tiny synthetic audio.

## 8. Engines and platforms

All engines sit behind one adapter interface (M3, M5). An adapter loads a model for a profile, reports the languages it supports, and transcribes a waveform into engine-neutral timed segments.

| Engine | Platform | Status |
|---|---|---|
| MLX Whisper | Apple silicon, native | First version. The Mac engine (M13) |
| faster-whisper | CUDA in Docker | First version. The CUDA engine, configured as at the source revision: `float32`, sequential (M13) |
| Azure Speech | Any, network | Planned, postponed (M6). Explicitly named profile only, never a fallback |
| TensorRT-LLM, whisper.cpp | CUDA, both | Not chosen: too early to decide (M13). A later decision may add one as a benchmark candidate behind the same adapter interface |
| WhisperKit | Apple | Out of scope: it is Swift and needs a process boundary |

- The default model is full Whisper `large-v3` (D53 at the source revision). Turbo and distilled weights are different models. A profile that uses them names them, and results never present them as `large-v3`.
- Within-file batching, such as faster-whisper's `BatchedInferencePipeline`, is a profile setting, measured as a separate profile.
- MLX is installed only on macOS. In the devcontainer, the MLX adapter is type-checked against a narrow stub or override and tested with a fake (M10).

## 9. Deployment and development environment

Service deployment:

- **CUDA:** one Docker Compose service runs one process with the NVIDIA runtime and `restart: unless-stopped`. It has a bind mount of the staging share, and volumes for the database and the model weights.
- **Apple:** one native macOS process runs from a uv-managed environment, and a supervisor restarts it (assumption A4). The database and model weights are in configured local directories outside the share. Docker on the Mac cannot use Metal, so the Mac service MUST NOT run in a container.
- Both are unauthenticated and LAN-only, as decision D5 at the source revision states. Neither is exposed to the public internet.
- The SQLite database is never on the SMB share.
- Model weights are downloaded once into the configured weights directory. Neither the service nor a test downloads weights implicitly during a job.

Development environment (M10):

- The default environment is the repository's devcontainer, as in `youtube-collector`. The gates run there. Its storage is development-only: no real staging share, no real model weights directory, no video library.
- A **host check** is work that needs a real accelerator or real host data, run outside the devcontainer. A Mac host check runs on the user's Mac, and a CUDA host check runs on the CUDA machine (M15).
- Host checks include the MLX and faster-whisper adapters' contract tests, every run of the service against the real share, every benchmark run, and every model download. Azure requests join them when Azure is added (M6).
- Each host check needs the user's authorization for that run, like the collector's runs against real YouTube. The authorization covers only the stated run.
- An agent that finds itself outside the devcontainer stops and confirms with the user, unless the user has authorized that host check.

This section is the one definition of the development environment and host checks. Section 11 and AGENTS refer to it.

Dependencies are locked with uv. `pyproject.toml` declares no upper bounds, and every full upgrade's `uv.lock` diff is reviewed, as in the collector's C32.

## 10. Behavior carried over and dropped

Carried over from `youtube-whisperer` at the source revision, reimplemented here:

| Behavior | Source |
|---|---|
| ffmpeg decoding to mono 16 kHz float32 | `transcriber/waveform_loader.py` |
| Failing on an empty waveform before model work | `whisper_transcriber.py` |
| Fully consuming lazy segments before publication | D22 and `whisper_transcriber.py` |
| Whole-run rejection on known bad phrases | `transcriber/rejection_policy.py` |
| Resident model kept across jobs | `get_default_whisper_model` |
| SRT formatting rules | `adaptors/srt_deduplicator.py` |
| `NEVER` preflight before model work, atomic staged publication | D48, D49; pathlib-extensions `StagedFile` |
| LAN-only, unauthenticated trust model | D5 |

Changed or dropped:

| Behavior at the source revision | Here |
|---|---|
| Redis Streams and the dead-letter list | SQLite job table and the failure resource, as in the collector |
| Fixed language whitelist and BCP to ISO conversion | Any tag the engine supports, mapped by the adapter |
| faster-whisper as the only local engine, CUDA `float32` or CPU `int8` | faster-whisper CUDA `float32` on NVIDIA, MLX Whisper on Apple (M13). No CPU production path |
| `OverwritePolicy.ALWAYS` on SRT helpers | `NEVER` only (M9) |
| Python entry points under D18 | Not kept. The HTTP API is the interface (assumption A1) |
| WAV sidecar conversion | Dropped. Decoding happens in memory |
| Azure transcriber | Postponed (M6) |
| SRT deduplication of caption downloads | Not part of the transcriber. Captions belong to the collector |

## 11. Engineering standards and governance

The repository adopts the governance of `youtube-collector`, which `ClipboardTTS` shares. That is AGENTS.md, STYLE.md, `tasks/` with TODO and completion records, `tests/README.md` and `project-reviews.md`. It also includes the adversarial review and grouped task execution skills, the shared `agent-workflow-common` tools through `agent-workflow.json`, and the coverage gate, mypy and ruff. Scaffolding them is the first task after this document is accepted.

Adaptations for this project:

- **Environment rule.** AGENTS adopts section 9's development environment and host checks (M10) in place of the collector's Docker-only rule.
- **Real host data.** The staging share, the real model weights directory, the video library that holds the benchmark recordings, and later the Azure credentials. Agents never point a development or test run at them.
- **Gates.** `./check-coverage.sh`, `mypy media_transcriber` and `ruff check .`, in the devcontainer. Platform-only engine code is covered there through fakes, and its real behavior through the host checks.
- **Isolation of tests.** No real models, accelerators or network. Engine adapters are tested against fakes. An optional contract test per real engine runs only as a host check.
- **Benchmark evidence.** Historical benchmark evidence states its revision, machine, recording hashes and limitations, and does not certify current results. This follows AGENTS' rule on facts.

## 12. Integration with the orchestrator and the collector

Switching the orchestrator over is a later, separate phase, as for the collector:

1. The orchestrator learns from the collector that an item's captions are `unavailable` or `error`, with the MP4's path relative to the staging root.
2. It submits that path, the language and optionally a profile to one deployment, and saves the job ID.
3. It polls until the job ends. A failure with category `missing` means the media file was not found. A likely cause is that the user moved it, but an unmounted or misconfigured share and a wrong path are also possible. The orchestrator records the failure and does not resubmit on its own.
4. It records the failure the job carries and does not clear it. A missing job (404) most likely means the job has passed its retention. A recreated database or a request to the wrong deployment gives the same answer, so the orchestrator reports it rather than assuming success.

For a video that needs transcription, the SRT appears in staging some time after its MP4. Until then, an MP4 without an SRT means transcription is pending or failed. The user decides when to move files out of staging. Moving an MP4 before its job runs is one way that job can fail with category `missing`.

## 13. Decisions agreed in discussion (2026-11-17)

| ID | Decision |
|---|---|
| M1 | The project has two purposes: a transcriber service that runs on Apple M-series or on CUDA, and a benchmark tool that measures the same Whisper model on the same audio. |
| M2 | The service follows `youtube-collector`'s shape: a LAN-only, unauthenticated HTTP service with its own SQLite queue, submission returning a job ID, and polling. This supersedes the redesign handoff's orchestrator-owned queue, as the collector's C1 did. |
| M3 | One Python codebase with one adapter per engine. It runs in Docker on CUDA and as a native macOS process on Apple silicon. Swift engines such as WhisperKit are out unless wrapped behind a process boundary later. |
| M4 | ffmpeg decoding, the rejection policy, the `NEVER` preflight and atomic staged SRT publication carry over from `youtube-whisperer`. |
| M5 | The benchmark is a re-runnable regression check at its core. Backend selection and machine comparison are uses of its stored results. It times the service's own engine adapters, so a candidate backend gets an adapter. Results from different machines are not presented as like-for-like hardware comparisons. |
| M6 | Azure Speech stays as an engine a caller can choose by name, postponed past the first version. It is never an automatic fallback. |
| M7 | The service reads media from, and writes the SRT to, the same SMB staging share the collector writes to. Requests use paths relative to the staging root, and each host mounts the share at its own path. The SRT is published beside its media. |
| M8 | A marker file of the transcriber's own guards against an unmounted share. Each process runs one job at a time with one resident model. Each deployment configures its engine profiles and a default. A request names the source language. |
| M9 | No re-transcription over an existing SRT: `NEVER` only, so no provenance sidecar is needed. |
| M10 | The devcontainer is the default development environment and runs the gates. Work that needs the Mac GPU or a CUDA GPU, including every benchmark run, is a host check that needs the user's authorization per run. |

Decisions on the proposals, made by the user later on 2026-11-17:

| ID | Decision |
|---|---|
| M11 | The benchmark's headline metric is wall time from the start of decoding to a complete SRT for one recording. Inference-only time is a diagnostic. |
| M12 | The benchmark measures time only. It computes no quality score, so the corpus needs no reference transcripts. |
| M13 | The engines are MLX Whisper on Apple silicon and faster-whisper on CUDA, as `youtube-whisperer` uses it, both with full `large-v3`. TensorRT-LLM was considered and set aside: it is too early to decide on it. Its Whisper path takes clips of at most 30 seconds without timestamps, so it would need its own splitting and timing. |

Decisions on the open questions, made by the user later on 2026-11-17:

| ID | Decision |
|---|---|
| M14 | Settles Q1 to Q3: the repository contains no benchmark recordings. The user keeps one English and one Chinese recording in the video library. The benchmark measures neither energy nor memory use, and no speed threshold is recorded. |
| M15 | Settles Q5: the Apple deployment is the user's M2 Mac. The CUDA machine is not available yet, so the CUDA deployment and its host checks wait until it is. |
| M16 | Settles Q6: no other engine is added for now. MLX Whisper and faster-whisper are the only engines (M13). |

Decision from the external review of revision `e6b4354`, applied at the user's request:

| ID | Decision |
|---|---|
| M17 | Health judges progress, not thread liveness, following the collector's C55 rule: the staleness limit applies only while the worker is idle or in an activity that touches the share (5.11). |

Decision on the open point that T1 recorded in T7, made by the user on 2026-11-17:

| ID | Decision |
|---|---|
| M18 | Refines M17 as `youtube-collector`'s T42 refines C55: during decoding and publication, the worker refreshes staging freshness only by checking the marker again between chunks. A progress report alone does not refresh it (5.11). |

Decisions made by the user when T3 and T6 were bounded, and after T2's checks of the staging share, on 2026-11-17 and 2026-11-18:

| ID | Decision |
|---|---|
| M19 | SRT serialization merges consecutive segments with identical text into one cue that spans them, as `youtube-whisperer`'s transcription path did at the source revision. This affects T3. |
| M20 | Publication requires that the media file still exists. If it is gone, the job fails with category `missing` and no SRT is written, so no SRT is left without its media and the transcriber never creates a directory. This affects T6 and T7. |
| M21 | Refines 5.6 and M9 after T2's checks showed that the Mac's SMB mount supports neither hard links nor an exclusive rename, and that a plain rename replaces files. Where hard links are unsupported, publication writes the complete SRT in the hidden stage, checks that no SRT exists, and renames the stage into place. The user accepts that an SRT created by another writer between the check and the rename is replaced. Only another transcription of the same media could create one. The behavior is added to pathlib-extensions. This affects T6, T7, T11 and the new T12. |

## 14. Open questions

| ID | Question |
|---|---|
| Q4 | Which parts of a result set are committed, if any. Numbers and environment records are small. Produced SRTs may contain copyrighted speech. The user considers it too early to decide. |

Q1 to Q3, Q5 and Q6 are settled as M14 to M16.

## 15. Assumptions

These follow from repository evidence or the discussion, and stand unless the user objects:

- **A1:** nothing in this project keeps backward compatibility with `youtube-whisperer`'s Python API. The orchestrator talks HTTP. API, settings and database change without migration, as in the collector's C44.
- **A2:** job retention, failure records, polling, restart recovery and crash handling follow the collector's sections 5.8 to 5.12, adapted to one job per submission. Health follows 5.11 and M17 instead of the collector's older C43 rule.
- **A3:** the Python version follows the collector (3.14) if faster-whisper, CTranslate2 and MLX Whisper publish wheels for it. Otherwise the project uses the newest version they all support. The builder verifies this before scaffolding.
- **A4:** the Mac supervisor is launchd, unless the builder finds a simpler equivalent. The service is started by the user, and the repository provides the configuration.
- **A5:** the tuning values (retention, grace periods, pause before exit, repetitions) are starting defaults, not measured requirements.

## 16. Architecture

This section records the modules as tasks build them. It describes the current code, not plans.

| Module | Responsibility | Task |
|---|---|---|
| `media_transcriber/engine.py` | `Segment`, `Profile`, and the `EngineAdapter` and `TranscriptionModel` protocols that every engine implements (section 8) | T3 |
| `media_transcriber/audio.py` | Decoding with an ffmpeg child process into a mono 16 kHz float32 waveform, read in chunks with a checkpoint between them (5.5 step 4, M18) | T3 |
| `media_transcriber/rejection.py` | Whole-run rejection of known bad phrases (5.5 step 7) | T3 |
| `media_transcriber/srt.py` | SRT serialization, with cues of identical text merged (5.6, M19) | T3 |
| `media_transcriber/pipeline.py` | `run_pipeline`: decoding, transcription, rejection and serialization with a loaded model. It returns the text and the stage timings, and never touches staging. The worker and the benchmark share it (5.5, 7.3) | T3 |
| `media_transcriber/staging.py` | Request path validation without filesystem access, containment in the staging root through symlinks, the marker check, the `NEVER` preflight, and atomic SRT publication through `StagedFile` that never creates a directory. Where hard links are unsupported, publication renames after a check (5.1, 5.2, 5.6, 5.10, M18, M20, M21) | T6, T12 |

Dataflow of one transcription: `run_pipeline` calls `decode_waveform`, then the model's `transcribe`. It consumes every segment through `reject_bad_segments` and then calls `segments_to_srt`. ffmpeg runs as a subprocess, not through `ffmpeg-python`, because the decoder needs chunked reads and owns the child's lifecycle.

Staging is separate from the pipeline. The worker (T7) will call `require_marker`, `resolve_job_paths` and `srt_exists` before the pipeline, and `publish_srt` after it, with the marker check as the checkpoint of both decoding and publication.
