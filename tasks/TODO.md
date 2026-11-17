# TODO — Open work

This file holds the recommended order, the constraints that several tasks share, and the index of open tasks. [README.md](README.md) owns how tasks are recorded, bounded and closed.

- Read [AGENTS.md](../AGENTS.md), [DESIGN.md](../DESIGN.md) and the relevant operating instructions in the [project README](../README.md) before you accept an implementation task.
- A recorded task is not permission to combine unrelated changes or to redesign the service.
- For a group of related tasks, use the [grouped task execution skill](../.agents/skills/grouped-task-execution/SKILL.md).

## Recommended order

The tasks below were recorded with the governance scaffold, [T1](closed/T1.md). They are compact until accepted, and each is bounded against DESIGN when it starts.

1. T5, the MLX Whisper adapter. [T3](closed/T3.md), the pipeline, [T6](closed/T6.md), the staging layer, and [T4](closed/T4.md), the faster-whisper adapter, are closed.
2. T7, then T8: the store and worker, then the HTTP API and the service entry point.
3. T9: the benchmark, after at least one adapter.

These tasks are outside the sequence:

- T2 collects host checks as tasks land.
- T10 waits for the CUDA machine (DESIGN M15).
- T11 follows T8.

## Shared constraints

A host check needs a Docker daemon, a real accelerator or real host data, and runs outside the devcontainer with the user's authorization for each run ([DESIGN section 9](../DESIGN.md#9-deployment-and-development-environment)). Collect host checks in [T2](open/T2.md).

## Open tasks

- [T2](open/T2.md) — Low. Host checks.
- [T5](open/T5.md) — High. MLX Whisper adapter.
- [T7](open/T7.md) — High. Store and worker.
- [T8](open/T8.md) — High. HTTP API and service entry point.
- [T9](open/T9.md) — Medium. Benchmark command and result sets.
- [T10](open/T10.md) — Medium, blocked on the CUDA machine. Production CUDA image and Compose service.
- [T11](open/T11.md) — Medium. Mac deployment with launchd.
