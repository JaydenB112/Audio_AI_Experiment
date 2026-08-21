"""
Step 1: bare-bones audio loop.

Confirms the Pipecat audio pipeline works end to end on your laptop before
any AI is involved. Captures audio from your mic and plays it straight back
out your speaker. Talk into the mic, you should hear your own voice come
back out with a small delay.

No STT, no Claude, no TTS yet. That comes in later steps.

Optionally reads MIC_DEVICE_NAME / SPEAKER_DEVICE_NAME from .env (see
list_audio_devices.py) to test specific hardware instead of the system
default mic/speaker. Handy for sanity-checking one new piece of hardware
(e.g. a speaker) before the rest of the setup (e.g. a mic) arrives.

Run:
    python step1_audio_loop.py

Stop with Ctrl+C.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

from audio_devices import find_device_index

load_dotenv()

logger.remove(0)
logger.add(sys.stderr, level="INFO")


class LoopbackProcessor(FrameProcessor):
    """Re-tags each captured mic frame as a speaker frame so the local audio
    output stage will play it. Every other frame (start, end, etc.) passes
    through untouched.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            echo_frame = OutputAudioRawFrame(
                audio=frame.audio,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )
            await self.push_frame(echo_frame, direction)
        else:
            await self.push_frame(frame, direction)


async def main():
    mic_name = os.environ.get("MIC_DEVICE_NAME")
    speaker_name = os.environ.get("SPEAKER_DEVICE_NAME")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # Pin input and output to the same rate. Pipecat's defaults differ
            # (16000 in, 24000 out), and this script plays captured audio back
            # unchanged, so a mismatch here speeds up and pitch-shifts it.
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            # Optional MIC_DEVICE_NAME / SPEAKER_DEVICE_NAME from .env, matched
            # by substring against list_audio_devices.py. Unset uses the
            # system default device.
            input_device_index=find_device_index(mic_name, want_input=True) if mic_name else None,
            output_device_index=(
                find_device_index(speaker_name, want_input=False) if speaker_name else None
            ),
        )
    )

    pipeline = Pipeline(
        [
            transport.input(),
            LoopbackProcessor(),
            transport.output(),
        ]
    )

    worker = PipelineWorker(pipeline, params=PipelineParams())

    runner = WorkerRunner()
    await runner.add_workers(worker)

    logger.info("Audio loop running. Talk into your mic, you should hear yourself in the speaker.")
    logger.info("Press Ctrl+C to stop.")

    await runner.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped.")
