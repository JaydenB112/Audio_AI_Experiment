"""
Step 6: swap in The Archive persona.

Same full loop as step 5: mic -> Deepgram STT -> Claude -> ElevenLabs TTS ->
speaker, with the interruption state log still in place. The only change is
the system prompt, which now speaks as The Archive instead of a generic
placeholder assistant.

Requires headphones, same reason as step 4.

Requires DEEPGRAM_API_KEY, ANTHROPIC_API_KEY, and ELEVENLABS_API_KEY in a
.env file (see .env.example). ELEVENLABS_VOICE_ID is optional, it defaults
to the "BVO" custom voice already set up in this ElevenLabs account.

Run:
    python step6_archive.py

Stop with Ctrl+C.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv()

logger.remove(0)
logger.add(sys.stderr, level="INFO")

ARCHIVE_SYSTEM_PROMPT = (
    "You are the voice of The Archive, a lore-keeper terminal from the BVO "
    "(Baron Von Opperbean and the River of Time) multiverse. You collect and speak news, "
    "sightings, and notices from across the realms, including Port of Prizmo, Platform 34, "
    "the Gizoku faction, and the Green Dragon. Speak like a fragment of a larger story, not "
    "a chatbot. Mysterious and inviting, never comedic or self-aware about being an AI. "
    "Keep responses short, a sentence or two, since this is spoken aloud to a visitor "
    "standing in front of you, not read on a screen."
)

# The "BVO" custom voice already in this ElevenLabs account. "BVO 2" and
# "BVO 3" also exist there if this one isn't the right take.
# Override with ELEVENLABS_VOICE_ID to use a different voice.
DEFAULT_VOICE_ID = "MlxqcnXQGRrE9KXbBdaJ"

SAMPLE_RATE = 16000


class ResponsePrinter(FrameProcessor):
    """Buffers streamed LLM text chunks and prints the full response once
    Claude finishes, then passes every frame through untouched.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._buffer = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = ""
        elif isinstance(frame, LLMTextFrame):
            self._buffer += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            logger.info(f"[archive] {self._buffer}")

        await self.push_frame(frame, direction)


class InterruptionMonitor(FrameProcessor):
    """Logs bot/user speaking state transitions so interruption handling can
    be confirmed from the console instead of by ear alone. Placed between
    the LLM and TTS stages so it sees UserStartedSpeakingFrame flowing
    downstream from the VAD-driven aggregator, and BotStartedSpeakingFrame /
    BotStoppedSpeakingFrame flowing upstream from the transport output.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bot_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            logger.info("[state] bot started speaking")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            logger.info("[state] bot stopped speaking")
        elif isinstance(frame, UserStartedSpeakingFrame):
            if self._bot_speaking:
                logger.warning("[state] user started speaking while bot was talking, interruption")
            else:
                logger.info("[state] user started speaking")
        elif isinstance(frame, InterruptionFrame):
            logger.warning("[state] interruption frame pushed")

        await self.push_frame(frame, direction)


async def main():
    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not deepgram_api_key or not anthropic_api_key or not elevenlabs_api_key:
        logger.error(
            "DEEPGRAM_API_KEY, ANTHROPIC_API_KEY, and ELEVENLABS_API_KEY must all be set. "
            "Copy .env.example to .env and fill them in."
        )
        return

    voice_id = os.environ.get("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE_ID

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
        )
    )

    stt = DeepgramSTTService(
        api_key=deepgram_api_key,
        settings=DeepgramSTTService.Settings(
            interim_results=True,
        ),
    )

    llm = AnthropicLLMService(
        api_key=anthropic_api_key,
        settings=AnthropicLLMService.Settings(
            model="claude-sonnet-5",
            system_instruction=ARCHIVE_SYSTEM_PROMPT,
        ),
    )

    tts = ElevenLabsTTSService(
        api_key=elevenlabs_api_key,
        sample_rate=SAMPLE_RATE,
        settings=ElevenLabsTTSService.Settings(
            voice=voice_id,
            model="eleven_flash_v2_5",
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            ResponsePrinter(),
            InterruptionMonitor(),
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(pipeline, params=PipelineParams())

    runner = WorkerRunner()
    await runner.add_workers(worker)

    logger.info("The Archive is listening.")
    logger.info("Press Ctrl+C to stop.")

    await runner.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped.")
