"""Full-duplex acoustic echo cancellation for Pipecat's local transport.

Pipecat's input-filter hook only receives microphone audio. A real echo
canceller also needs the far-end signal that is being played through the
speaker, so this module pairs a ``BaseAudioFilter`` with a small local output
transport subclass. The output side supplies the reference; the input side
runs WebRTC AEC3 before audio reaches VAD or STT.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from loguru import logger
from pywebrtc_audio import AudioProcessor

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import (
    FilterControlFrame,
    FilterEnableFrame,
    FilterUpdateSettingsFrame,
    OutputAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.local.audio import (
    LocalAudioOutputTransport,
    LocalAudioTransport,
    LocalAudioTransportParams,
)


class WebRTCAECFilter(BaseAudioFilter):
    """Pipecat input filter backed by WebRTC's AEC3 implementation.

    Audio is signed 16-bit mono PCM, which is also the format used by
    ``LocalAudioTransport`` in this project. Speaker reference bytes are
    consumed at the microphone's real-time rate; missing reference audio is
    represented by silence so user speech remains available during playback.
    """

    def __init__(self, *, stream_delay_ms: int = 40):
        self._stream_delay_ms = stream_delay_ms
        self._sample_rate: int | None = None
        self._processor: AudioProcessor | None = None
        self._reference_chunks: deque[bytes] = deque()
        self._reference_offset = 0
        self._reference_bytes = 0
        self._enabled = True

    async def start(self, sample_rate: int):
        if sample_rate not in (16000, 32000, 48000):
            raise ValueError(
                f"WebRTC AEC requires 16000, 32000, or 48000 Hz; got {sample_rate}"
            )

        self._sample_rate = sample_rate
        self._processor = AudioProcessor(
            sample_rate=sample_rate,
            num_channels=1,
            echo_cancellation=True,
            stream_delay_ms=self._stream_delay_ms,
        )
        self._clear_reference()
        logger.info(
            f"[aec] WebRTC AEC3 enabled at {sample_rate} Hz "
            f"(stream delay {self._stream_delay_ms} ms)"
        )

    async def stop(self):
        self._processor = None
        self._sample_rate = None
        self._clear_reference()

    async def process_frame(self, frame: FilterControlFrame):
        if isinstance(frame, FilterEnableFrame):
            self._enabled = frame.enable
            self._clear_reference()
            if self._processor:
                self._processor.reset()
        elif isinstance(frame, FilterUpdateSettingsFrame):
            delay = frame.settings.get("stream_delay_ms")
            if delay is not None:
                self._stream_delay_ms = int(delay)
                if self._processor:
                    self._processor.stream_delay_ms = self._stream_delay_ms

    def add_speaker_reference(self, audio: bytes) -> None:
        """Queue PCM that is about to be written to the speaker device."""
        if not self._enabled or not self._processor or not audio:
            return

        self._reference_chunks.append(audio)
        self._reference_bytes += len(audio)

        # A large backlog means timing has become stale. Bound it to two
        # seconds so AEC can recover instead of matching against old speech.
        max_bytes = (self._sample_rate or 16000) * 2 * 2
        while self._reference_bytes > max_bytes and len(self._reference_chunks) > 1:
            dropped = self._reference_chunks.popleft()
            self._reference_bytes -= len(dropped) - self._reference_offset
            self._reference_offset = 0

    async def filter(self, audio: bytes) -> bytes:
        if not self._enabled or not self._processor or not audio:
            return audio

        far = self._take_reference(len(audio))
        near_samples = np.frombuffer(audio, dtype=np.int16)
        far_samples = np.frombuffer(far, dtype=np.int16)
        clean_samples = self._processor.process(near_samples, far_samples)
        return np.asarray(clean_samples, dtype=np.int16).tobytes()

    def _take_reference(self, size: int) -> bytes:
        output = bytearray()
        while len(output) < size and self._reference_chunks:
            chunk = self._reference_chunks[0]
            available = len(chunk) - self._reference_offset
            take = min(size - len(output), available)
            output.extend(chunk[self._reference_offset : self._reference_offset + take])
            self._reference_offset += take
            self._reference_bytes -= take

            if self._reference_offset == len(chunk):
                self._reference_chunks.popleft()
                self._reference_offset = 0

        if len(output) < size:
            output.extend(bytes(size - len(output)))
        return bytes(output)

    def _clear_reference(self) -> None:
        self._reference_chunks.clear()
        self._reference_offset = 0
        self._reference_bytes = 0


class AECReferenceOutputTransport(LocalAudioOutputTransport):
    """Local output transport that sends played PCM to the AEC filter."""

    def __init__(self, py_audio, params, aec_filter: WebRTCAECFilter):
        super().__init__(py_audio, params)
        self._aec_filter = aec_filter

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        # Feed the reference immediately before handing the same bytes to
        # PyAudio. This includes only audio the transport actually attempts
        # to play, including silence generated during shutdown.
        if self._out_stream:
            self._aec_filter.add_speaker_reference(frame.audio)
        return await super().write_audio_frame(frame)


class AECLocalAudioTransport(LocalAudioTransport):
    """``LocalAudioTransport`` with a far-end-aware WebRTC input filter."""

    def __init__(
        self, params: LocalAudioTransportParams, aec_filter: WebRTCAECFilter
    ):
        if params.audio_in_filter is not aec_filter:
            raise ValueError("params.audio_in_filter must be the supplied AEC filter")
        super().__init__(params)
        self._aec_filter = aec_filter

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = AECReferenceOutputTransport(
                self._pyaudio, self._params, self._aec_filter
            )
        return self._output
