"""
Step 3: add Claude as the reasoning step.

Captures mic audio, runs it through Deepgram STT, and once you stop talking
sends the transcript to Claude with a placeholder system prompt. Prints
Claude's text response to console. Still no TTS and no speaker output, that
comes in step 4.

Requires DEEPGRAM_API_KEY and ANTHROPIC_API_KEY in a .env file (see
.env.example).

Run:
    python step3_claude.py

Stop with Ctrl+C.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
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
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv()

logger.remove(0)
logger.add(sys.stderr, level="INFO")

# Placeholder system prompt. Gets replaced with The Archive persona in step 6.
PLACEHOLDER_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep your responses short, "
    "a sentence or two, since they will be spoken aloud."
)


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
            logger.info(f"[claude] {self._buffer}")

        await self.push_frame(frame, direction)


async def main():
    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not deepgram_api_key or not anthropic_api_key:
        logger.error(
            "DEEPGRAM_API_KEY and ANTHROPIC_API_KEY must both be set. "
            "Copy .env.example to .env and fill them in."
        )
        return

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=False,
            audio_in_sample_rate=16000,
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
            system_instruction=PLACEHOLDER_SYSTEM_PROMPT,
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
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(pipeline, params=PipelineParams())

    runner = WorkerRunner()
    await runner.add_workers(worker)

    logger.info("Listening. Talk, then stop, Claude's reply prints below.")
    logger.info("Press Ctrl+C to stop.")

    await runner.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped.")
