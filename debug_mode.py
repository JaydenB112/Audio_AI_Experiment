"""
Debug mode: plain pipeline, no persona.

Same audio signal path as step7_sentinel.py/step8_presence.py (mic -> STT ->
Claude -> TTS -> speaker, with interruption logging), but with a minimal
system prompt instead of the Sentinel persona. For hardware/pipeline
debugging: confirms transcription and playback are working without waiting
through in-character deflections, refusals, or the "no outside-world
knowledge" restrictions, which actively get in the way when all you want is
a quick "yep, heard you" while testing mic/speaker setups.

Not a replacement for step7/step8, just a faster loop for hardware testing.
Swap back to step7_sentinel.py or step8_presence.py once you're testing the
actual character instead of the audio path.

Requires headphones (or the same speaker-into-your-ears setup as the mic)
unless you've confirmed the real hardware handles echo on its own.

Requires DEEPGRAM_API_KEY, ANTHROPIC_API_KEY, and ELEVENLABS_API_KEY in a
.env file (see .env.example). Reads the same MIC_DEVICE_INDEX,
SPEAKER_DEVICE_INDEX, and ELEVENLABS_VOICE_ID as the other steps.

Run:
    python debug_mode.py

Stop with Ctrl+C.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.workers.runner import WorkerRunner
from step7_sentinel import (
    DEFAULT_VOICE_ID,
    SAMPLE_RATE,
    InterruptionMonitor,
    ResponsePrinter,
    build_local_transport,
    check_required_env,
)

load_dotenv()

logger.remove()
logger.add(sys.stderr, level="INFO")

DEBUG_SYSTEM_PROMPT = """
You are a plain diagnostic assistant used to test a voice pipeline's audio
hardware. You have no persona and no character. When you receive a
message, respond with one short, direct sentence: briefly acknowledge what
you heard so the tester can confirm transcription worked. Do not roleplay,
do not refuse anything, do not add commentary, disclaimers, or personality.
Answer plainly and factually if asked something, the same as a normal
helpful assistant would.
"""


def build_debug_worker(transport) -> PipelineWorker:
    deepgram_api_key = os.environ["DEEPGRAM_API_KEY"]
    anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
    elevenlabs_api_key = os.environ["ELEVENLABS_API_KEY"]
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE_ID

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
            system_instruction=DEBUG_SYSTEM_PROMPT,
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

    return PipelineWorker(pipeline, params=PipelineParams())


async def main():
    check_required_env()

    transport = build_local_transport()
    worker = build_debug_worker(transport)

    runner = WorkerRunner()
    await runner.add_workers(worker)

    logger.info("DEBUG MODE. No persona, plain acknowledgments only.")
    logger.info("Press Ctrl+C to stop.")

    await runner.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped.")
