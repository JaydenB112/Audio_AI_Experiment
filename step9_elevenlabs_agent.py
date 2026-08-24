"""
Step 9: one presence-triggered conversation, using the ElevenLabs
Conversational AI Agents platform instead of the local Deepgram/Claude/
ElevenLabs-TTS pipeline in step7_sentinel.py.

The Agent (configured in the ElevenLabs dashboard: persona, voice, LLM, and
its own greeting) handles STT, the LLM turn, and TTS entirely on
ElevenLabs' side -- conversation history lives in the ElevenLabs dashboard,
not locally. This process only supplies full-duplex audio (through
ElevenLabsAudioInterface, which layers this project's WebRTC AEC and
named-device selection on top of the SDK's raw PyAudio streams) and the
presence-triggered state machine that starts and ends each session.

State flow, identical to step8_presence.py:
    SLEEPING -> (presence detected) -> AWAKENING -> conversation running
    -> (presence lost for DEPARTURE_GRACE_SECONDS, or the Agent ends the
    session on its own) -> process exits

Requires ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID in a .env file (see
.env.example). The Agent must already be configured in the ElevenLabs
dashboard with its own first-message/greeting behavior -- this script does
not send one, only the departure farewell.

Run:
    python step9_elevenlabs_agent.py

Stop with Ctrl+C.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from loguru import logger

from elevenlabs_audio_interface import ElevenLabsAudioInterface
from presence_sensor import KeystrokePresenceSensor, PresenceSensor

load_dotenv()

logger.remove()
logger.add(sys.stderr, level="INFO")

DEPARTURE_GRACE_SECONDS = 15.0
DEPARTURE_POLL_SECONDS = 1.0
FAREWELL_TIMEOUT_SECONDS = 15.0

DEPARTURE_CUE = (
    "(The traveler is leaving and will no longer be able to hear you. "
    "Acknowledge their departure briefly, in character, then fall silent.)"
)


def check_required_env() -> None:
    required = ["ELEVENLABS_API_KEY", "ELEVENLABS_AGENT_ID"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )


def build_conversation(audio_interface: ElevenLabsAudioInterface) -> Conversation:
    """Build a fresh Conversation for one visitor. Unlike a Pipecat
    transport, nothing here needs to be reused across conversations -- the
    SDK opens and closes its own websocket, and ElevenLabsAudioInterface
    its own PyAudio streams, around each session.
    """
    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    return Conversation(
        client=client,
        agent_id=os.environ["ELEVENLABS_AGENT_ID"],
        requires_auth=True,
        audio_interface=audio_interface,
        callback_end_session=lambda: logger.info("[state] ElevenLabs ended the session"),
    )


async def watch_for_departure(
    sensor: PresenceSensor,
    conversation: Conversation,
    audio_interface: ElevenLabsAudioInterface,
) -> None:
    """Poll the sensor while a conversation is running. Once it reports
    absence continuously for DEPARTURE_GRACE_SECONDS, have the Agent
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
            audio_interface.reset_idle()
            await asyncio.to_thread(conversation.send_user_message, DEPARTURE_CUE)
            if not await asyncio.to_thread(
                audio_interface.wait_until_idle, FAREWELL_TIMEOUT_SECONDS
            ):
                logger.warning("[state] timed out waiting for farewell to finish playing")
            logger.info("[state] ending conversation")
            await asyncio.to_thread(conversation.end_session)
            return


async def run_conversation(sensor: PresenceSensor) -> None:
    """Run one full conversation session: open the Agent connection, race
    it against presence loss, and clean up.
    """
    audio_interface = ElevenLabsAudioInterface()
    conversation = build_conversation(audio_interface)

    await asyncio.to_thread(conversation.start_session)

    session_task = asyncio.create_task(
        asyncio.to_thread(conversation.wait_for_session_end)
    )
    departure_task = asyncio.create_task(
        watch_for_departure(sensor, conversation, audio_interface)
    )

    done, _ = await asyncio.wait(
        {session_task, departure_task}, return_when=asyncio.FIRST_COMPLETED
    )

    if departure_task in done:
        # Departure already ended the session. Let session_task observe
        # that and finish on its own.
        await session_task
    else:
        # The Agent ended the conversation some other way. Stop watching
        # for departure and surface anything session_task raised.
        departure_task.cancel()
        try:
            await departure_task
        except asyncio.CancelledError:
            pass
        await session_task


async def run_once(sensor: PresenceSensor) -> None:
    logger.info("[state] SLEEPING")
    await sensor.wait_until_present()

    logger.info("[state] AWAKENING")
    await run_conversation(sensor)
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
