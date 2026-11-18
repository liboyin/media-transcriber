"""Run the service: `python3 -m media_transcriber` (DESIGN section 9).

One uvicorn process serves the API and runs the worker thread. It must never run several server
processes, because the worker, its resident model and the values that `/health` reads live in
the process. Settings come from the environment (`config.Settings`). On SIGTERM uvicorn stops
serving, and the application's lifespan stops the worker, waiting up to the shutdown grace for it.
"""

import logging
import os
from collections.abc import Mapping

import uvicorn

from media_transcriber.api import create_app
from media_transcriber.config import Settings

__all__ = ['main']

REQUEST_SHUTDOWN_SECONDS = 5
"""How long uvicorn lets open requests finish after SIGTERM before it starts the application's shutdown."""


def main(environ: Mapping[str, str] = os.environ) -> None:
    """Read the settings, configure logging, and serve until stopped.

    Raises:
        config.SettingsError: If a setting or the profiles file is missing or invalid.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    settings = Settings.from_environ(environ)
    uvicorn.run(
        # workers=1 overrides an inherited WEB_CONCURRENCY, which would make uvicorn refuse an app instance.
        create_app(settings), host=settings.host, port=settings.port, workers=1,
        timeout_graceful_shutdown=REQUEST_SHUTDOWN_SECONDS, log_config=None,
    )


if __name__ == '__main__':
    main()
