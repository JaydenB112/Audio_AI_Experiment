"""
Step 8: one presence-triggered conversation.

Waits for the presence sensor, runs one Sentinel conversation, then exits.
Keeping each conversation in a fresh process ensures PyAudio, network
clients, and DSP state are fully released between visitors. For unattended
all-day operation, run convention_runner.py, which starts a fresh copy of
this script after each conversation.

State flow:
    SLEEPING -> (presence detected) -> AWAKENING -> conversation running
    -> (presence lost for DEPARTURE_GRACE_SECONDS, or the conversation ends
    on its own) -> process exits

Uses the real motion sensor via KeystrokePresenceSensor (presence_sensor.py)
-- it's a USB device that emulates a keyboard, sending 't' (talk, start the
conversation), 'q' (quit, end it), or 'r' (recognize, logged but not acted
on yet). ConsolePresenceSensor (press Enter to simulate arrival/departure)
is still in presence_sensor.py if you need to test the state machine
without the sensor plugged in -- swap it back in below for that.

Uses WebRTC AEC3 to remove the known speaker signal from microphone input,
so it can run through speakers while preserving interruption handling.

Requires DEEPGRAM_API_KEY, ANTHROPIC_API_KEY, and ELEVENLABS_API_KEY in a
.env file (see .env.example).

Run:
    python step8_presence.py

Stop with Ctrl+C.
"""

import asyncio
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.workers.runner import WorkerRunner
from presence_sensor import KeystrokePresenceSensor, PresenceSensor
from step7_sentinel import (
    DEPARTURE_CUE,
    GREETING_CUE,
    SentinelSession,
    announce,
    build_local_transport,
    build_sentinel_worker,
    check_required_env,
)

load_dotenv()

logger.remove()
logger.add(sys.stderr, level="INFO")

DEPARTURE_GRACE_SECONDS = 15.0
DEPARTURE_POLL_SECONDS = 1.0


async def watch_for_departure(sensor: PresenceSensor, session: SentinelSession) -> None:
    """Poll the sensor while a conversation is running. Once it reports
    absence continuously for DEPARTURE_GRACE_SECONDS, have the Sentinel
    acknowledge the departure, wait for that to finish being spoken, then
    end the conversation. A brief step away from the sensor doesn't end
    the conversation, only a sustained absence does.
    """
    absent_since: float | None = None
    loop = asyncio.get_running_loop()

    while True:
        await asyncio.sleep(DEPARTURE_POLL_SECONDS)

        if sensor.is_present():
            absent_since = None
            continue

        if absent_since is None:
            absent_since = loop.time()
            continue

        if loop.time() - absent_since >= DEPARTURE_GRACE_SECONDS:
            logger.info("[state] presence lost, saying goodbye")
            await announce(session, DEPARTURE_CUE)
            logger.info("[state] ending conversation")
            await session.worker.stop_when_done()
            return


async def run_conversation(sensor: PresenceSensor, transport) -> None:
    """Run one full conversation session against the shared transport:
    build a fresh pipeline, greet the traveler, run until it ends, and
    clean up.
    """
    session = build_sentinel_worker(transport)
    runner = WorkerRunner()
    await runner.add_workers(session.worker)

    # add_workers() only registers the worker -- the pipeline doesn't
    # actually start running (StartFrame propagating, STT/TTS connecting)
    # until runner.run() is awaited. Start that as a background task before
    # announce() queues the greeting, or the greeting sits in an unstarted
    # pipeline's queue until announce()'s own wait times out.
    session_task = asyncio.create_task(runner.run())

    await announce(session, GREETING_CUE)

    departure_task = asyncio.create_task(watch_for_departure(sensor, session))

    done, _ = await asyncio.wait(
        {session_task, departure_task}, return_when=asyncio.FIRST_COMPLETED
    )

    if departure_task in done:
        # Departure already spoke the farewell and queued an EndFrame. Let
        # the session finish shutting down on its own, don't cancel it
        # mid-cleanup.
        await session_task
    else:
        # The conversation ended some other way. Stop watching for
        # departure and surface anything session_task raised.
        departure_task.cancel()
        try:
            await departure_task
        except asyncio.CancelledError:
            pass
        await session_task

async def run_once(sensor: PresenceSensor) -> None:
    transport = build_local_transport()

    logger.info("[state] SLEEPING")
    await sensor.wait_until_present()

    logger.info("[state] AWAKENING")
    await run_conversation(sensor, transport)
    logger.info("[state] conversation complete; exiting")


async def main():
    check_required_env()

    sensor = KeystrokePresenceSensor()
    logger.info(
        "Listening for the motion sensor's keystrokes (t = talk/start, "
        "q = quit/end, r = recognize/logged only). Requires this process "
        "to have Input Monitoring permission on macOS."
    )

    await run_once(sensor)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped.")
