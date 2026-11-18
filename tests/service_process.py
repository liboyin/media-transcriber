"""A real service process with a fake engine, which the lifecycle tests in `test_main.py` start and stop.

Run as `python3 service_process.py CONTROL_DIR` with the service's usual environment. It goes through the real
entry point, replacing only the engine adapter and the store's clock, so it loads no model. It installs the
network guard for its whole life and reports each refused attempt in its output. Files in the control directory
are how a test steers it from another process:

- While `hold-transcription` exists, a transcription waits (`transcription-entered` announces it).
- While `fail-store` exists, every store transaction on the worker thread fails, which ends the worker.

Each transcription appends a line to `transcriptions`. The fatal pause is shortened, so a failed worker ends the
process quickly.
"""

import dataclasses
import sqlite3
import sys
import threading
import time
from pathlib import Path

import network_guard
from fake_engine import FakeAdapter, FakeModel

import media_transcriber
import media_transcriber.__main__ as entrypoint
from media_transcriber import faster_whisper_adapter
from media_transcriber.api import Components
from media_transcriber.engine import Segment

# Shortened so that a stalled request does not cost each test uvicorn's full 5 seconds.
REQUEST_SHUTDOWN_SECONDS = 0.2
FATAL_PAUSE_SECONDS = 0.2
POLL_SECONDS = 0.01
TEXT = 'hello'


def controlled_model(control: Path) -> FakeModel:
    """Return a model whose transcriptions are counted and can be held."""
    model = FakeModel([Segment(0.0, 0.1, TEXT)])

    def hold() -> None:
        with (control / 'transcriptions').open('a') as transcriptions:
            transcriptions.write('transcription\n')
        (control / 'transcription-entered').touch()
        while (control / 'hold-transcription').exists():
            time.sleep(POLL_SECONDS)

    model.on_transcribe.append(hold)
    return model


def controlled_clock(control: Path):
    """Return the store's clock, which fails on the worker thread while `fail-store` exists.

    Every store transaction reads this clock, so failing here fails a real transaction.
    """

    def clock() -> float:
        if threading.current_thread().name == 'transcriber-worker' and (control / 'fail-store').exists():
            raise sqlite3.OperationalError('injected failure: fail-store')
        return time.time()

    return clock


def main() -> None:
    """Guard the process, then run the real entry point with a fake engine, printing where the package came from."""
    network_guard.guard_process()
    control = Path(sys.argv[1])
    print(f'PACKAGE {media_transcriber.__file__}', flush=True)
    adapter = FakeAdapter(name=faster_whisper_adapter.ENGINE, model=controlled_model(control))
    components = Components(adapters={adapter.name: adapter}, clock=controlled_clock(control))
    real_create_app = entrypoint.create_app
    entrypoint.create_app = lambda settings: real_create_app(
        dataclasses.replace(settings, fatal_pause=FATAL_PAUSE_SECONDS), lambda _settings: components,
    )
    entrypoint.REQUEST_SHUTDOWN_SECONDS = REQUEST_SHUTDOWN_SECONDS  # type: ignore[misc]
    entrypoint.main()


if __name__ == '__main__':
    main()
