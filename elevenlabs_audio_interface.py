"""
Custom AudioInterface for the ElevenLabs Conversational AI SDK.

The SDK's own DefaultAudioInterface just opens the system's default PyAudio
devices with no echo cancellation. This one layers this project's WebRTC
AEC (aec.py) and name-matched device selection (audio_devices.py) on top,
so the ElevenLabs Agent gets the same full-duplex echo cancellation the
Pipecat-based pipeline in step7_sentinel.py already has -- needed because
the robot runs through open speakers, not headphones.

See elevenlabs.conversational_ai.conversation.AudioInterface for the four
methods (start, stop, output, interrupt) any audio interface must
implement; DefaultAudioInterface in that same package is the reference
PyAudio-only implementation this one is modeled on.
"""

import os
import queue
import threading

import pyaudio
from elevenlabs.conversational_ai.conversation import AudioInterface
from loguru import logger

from aec import WebRTCAECFilter
from audio_devices import find_device_index

SAMPLE_RATE = 16000  # required by the ElevenLabs Conversational AI websocket
INPUT_FRAMES_PER_BUFFER = 4000  # 250ms @ 16kHz, matches DefaultAudioInterface
OUTPUT_FRAMES_PER_BUFFER = 1000  # 62.5ms @ 16kHz, matches DefaultAudioInterface


def _drive(coro):
    """Run a coroutine that never actually suspends, synchronously.

    WebRTCAECFilter's start/stop/filter methods are declared async only to
    satisfy Pipecat's BaseAudioFilter interface -- the actual work
    (pywebrtc_audio's AudioProcessor) is a plain synchronous call with no
    real await inside. Driving them this way lets this PyAudio-callback-
    driven interface reuse that filter without spinning up an event loop
    just to run code that never yields.
    """
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value
    raise RuntimeError("expected coroutine to complete without suspending")


class ElevenLabsAudioInterface(AudioInterface):
    """Full-duplex mic/speaker interface with WebRTC AEC for the ElevenLabs
    Conversational AI SDK. The Conversation object opens and closes one of
    these per session (start()/stop() bracket a single conversation), so
    build a fresh instance for every visitor.
    """

    def __init__(self, *, stream_delay_ms: int | None = None):
        mic_name = os.environ.get("MIC_DEVICE_NAME")
        speaker_name = os.environ.get("SPEAKER_DEVICE_NAME")
        self._input_device_index = (
            find_device_index(mic_name, want_input=True) if mic_name else None
        )
        self._output_device_index = (
            find_device_index(speaker_name, want_input=False) if speaker_name else None
        )
        if stream_delay_ms is None:
            stream_delay_ms = int(os.environ.get("AEC_STREAM_DELAY_MS", "40"))
        self._aec = WebRTCAECFilter(stream_delay_ms=stream_delay_ms)

        self._pyaudio = pyaudio.PyAudio()
        self._input_callback = None
        self._in_stream = None
        self._out_stream = None
        self._output_queue: queue.Queue[bytes] = queue.Queue()
        self._should_stop = threading.Event()
        self._output_thread: threading.Thread | None = None

        # Set whenever nothing is queued or playing. Cleared by output()
        # and by reset_idle(), which callers use to guard against
        # wait_until_idle() returning immediately for a turn that hasn't
        # started producing audio yet.
        self._idle = threading.Event()
        self._idle.set()

    def start(self, input_callback) -> None:
        self._input_callback = input_callback
        _drive(self._aec.start(SAMPLE_RATE))

        self._should_stop.clear()
        self._output_thread = threading.Thread(target=self._run_output, daemon=True)

        self._in_stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self._input_device_index,
            stream_callback=self._on_input,
            frames_per_buffer=INPUT_FRAMES_PER_BUFFER,
            start=True,
        )
        self._out_stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            output=True,
            output_device_index=self._output_device_index,
            frames_per_buffer=OUTPUT_FRAMES_PER_BUFFER,
            start=True,
        )
        self._output_thread.start()
        logger.info("[audio] mic/speaker streams open, AEC enabled")

    def stop(self) -> None:
        self._should_stop.set()
        if self._output_thread:
            self._output_thread.join()
        if self._in_stream:
            self._in_stream.stop_stream()
            self._in_stream.close()
        if self._out_stream:
            self._out_stream.close()
        self._pyaudio.terminate()
        _drive(self._aec.stop())
        logger.info("[audio] mic/speaker streams closed")

    def output(self, audio: bytes) -> None:
        self._idle.clear()
        self._output_queue.put(audio)

    def interrupt(self) -> None:
        try:
            while True:
                self._output_queue.get_nowait()
        except queue.Empty:
            pass

    def reset_idle(self) -> None:
        """Clear the idle signal ahead of triggering a cue whose spoken
        reply the caller intends to wait for, so wait_until_idle() blocks
        for that reply instead of returning immediately because nothing
        happened to be playing at the moment it was called. Mirrors
        InterruptionMonitor.reset_bot_stopped_event() in step7_sentinel.py.
        """
        self._idle.clear()

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        """Block until all queued output audio has finished playing.
        Returns False on timeout. Used to know when a triggered cue (the
        departure farewell) has actually been spoken, the same role
        InterruptionMonitor.wait_for_bot_stopped() plays in
        step7_sentinel.py.
        """
        return self._idle.wait(timeout=timeout)

    def _on_input(self, in_data, frame_count, time_info, status):
        clean = _drive(self._aec.filter(in_data))
        if self._input_callback:
            self._input_callback(clean)
        return (None, pyaudio.paContinue)

    def _run_output(self) -> None:
        while not self._should_stop.is_set():
            try:
                audio = self._output_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            self._aec.add_speaker_reference(audio)
            self._out_stream.write(audio)
            if self._output_queue.empty():
                self._idle.set()
