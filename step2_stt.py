"""
Step 2: add Deepgram STT to the pipeline.

Captures mic audio and runs it through Deepgram speech-to-text. Prints both
interim (partial, still-changing) and final transcripts to console so you
can judge accuracy and latency before anything downstream gets added. No
speaker output yet, no Claude, no TTS.

Requires DEEPGRAM_API_KEY in a .env file (see .env.example).

Run:
    python step2_stt.py

Stop with Ctrl+C.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv()

logger.remove(0)
logger.add(sys.stderr, level="INFO")


class TranscriptPrinter(FrameProcessor):
    """Logs interim and final transcripts as they arrive, then passes every
    frame through untouched.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            logger.info(f"[final]   {frame.text}")
        elif isinstance(frame, InterimTranscriptionFrame):
            logger.info(f"[interim] {frame.text}")

        await self.push_frame(frame, direction)


async def main():
    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not deepgram_api_key:
        logger.error("DEEPGRAM_API_KEY is not set. Copy .env.example to .env and fill it in.")
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

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            TranscriptPrinter(),
        ]
    )

    worker = PipelineWorker(pipeline, params=PipelineParams())

    runner = WorkerRunner()
    await runner.add_workers(worker)

    logger.info("Listening. Talk into your mic, transcripts will print below.")
    logger.info("Press Ctrl+C to stop.")

    await runner.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped.")
