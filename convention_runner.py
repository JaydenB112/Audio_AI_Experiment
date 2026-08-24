"""Run one fresh Step 8 process per convention visitor.

This supervisor deliberately stays small: the child owns all audio, sensor,
Pipecat, and API resources, then the operating system reclaims them when the
conversation ends. A failed child is restarted with bounded exponential
backoff so a persistent hardware or configuration problem cannot create a
tight restart loop.

Run:
    python convention_runner.py

Stop the supervisor and its current conversation with Ctrl+C.
"""

import asyncio
import sys
from pathlib import Path

from loguru import logger

STEP8 = Path(__file__).with_name("step8_presence.py")
NORMAL_RESTART_DELAY_SECONDS = 2.0
MAX_FAILURE_BACKOFF_SECONDS = 60.0


async def run_convention() -> None:
    failure_backoff = NORMAL_RESTART_DELAY_SECONDS

    while True:
        logger.info("[supervisor] starting a fresh conversation process")
        process = await asyncio.create_subprocess_exec(sys.executable, str(STEP8))

        try:
            return_code = await process.wait()
        except asyncio.CancelledError:
            if process.returncode is None:
                logger.info("[supervisor] stopping current conversation")
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning("[supervisor] child did not stop; killing it")
                    process.kill()
                    await process.wait()
            raise

        if return_code == 0:
            failure_backoff = NORMAL_RESTART_DELAY_SECONDS
            delay = NORMAL_RESTART_DELAY_SECONDS
            logger.info("[supervisor] conversation ended normally")
        else:
            delay = failure_backoff
            failure_backoff = min(failure_backoff * 2, MAX_FAILURE_BACKOFF_SECONDS)
            logger.error(
                f"[supervisor] child exited with status {return_code}; "
                f"retrying in {delay:.0f}s"
            )

        await asyncio.sleep(delay)


def main() -> None:
    try:
        asyncio.run(run_convention())
    except KeyboardInterrupt:
        logger.info("[supervisor] stopped")


if __name__ == "__main__":
    main()
