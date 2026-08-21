"""
Presence sensor abstraction.

Defines the interface a real sensor needs to implement, plus a
keyboard-driven stand-in for testing the state machine before any hardware
is wired up. Implement PresenceSensor against real hardware later (a GPIO
pin read for a PIR/mmWave sensor, or parsing a serial data stream) without
touching anything in step8_presence.py.
"""

import threading
from abc import ABC, abstractmethod

import asyncio
from pynput import keyboard
from loguru import logger


class PresenceSensor(ABC):
    """Interface for anything that can report whether someone is present."""

    @abstractmethod
    def is_present(self) -> bool:
        """Return the sensor's current reading. Must not block."""
        raise NotImplementedError

    async def wait_until_present(self, poll_interval: float = 0.5) -> None:
        """Block until is_present() becomes True. Polls by default; override
        if the underlying hardware can push an event instead.
        """
        while not self.is_present():
            await asyncio.sleep(poll_interval)


class ConsolePresenceSensor(PresenceSensor):
    """Keyboard stand-in for a real sensor. Press Enter in the terminal to
    toggle between present and absent, starting absent.
    """

    def __init__(self):
        self._present = False
        self._lock = threading.Lock()
        thread = threading.Thread(target=self._read_loop, daemon=True)
        thread.start()

    def _read_loop(self):
        while True:
            input()
            with self._lock:
                self._present = not self._present
                state = "PRESENT" if self._present else "ABSENT"
            print(f"[sensor] toggled to {state}")

    def is_present(self) -> bool:
        with self._lock:
            return self._present


class KeystrokePresenceSensor(PresenceSensor):
    """Real sensor: a USB motion sensor that emulates a keyboard, sending
    'r' (recognize, noticed someone at range), 't' (talk, close enough to
    start a conversation), or 'q' (quit, they've left).

    't' (talk) is what flips is_present() to True, since that's the signal
    that means "start the conversation." 'q' flips it back to False, same
    as the departure logic already expects. 'r' is logged but doesn't
    change presence, since there's nothing to plug it into yet (would need
    a lighting/head-turn system to build the two-zone "notice, then
    engage" experience the sensor is clearly designed for). If 'r' should
    also start conversations, add it alongside 't' below.

    Uses a global keyboard listener (pynput), not input(), since the
    sensor's keystrokes need to be seen regardless of which window (if
    any) has focus -- unlike ConsolePresenceSensor, which only works while
    the terminal running this script is focused. On macOS this requires
    granting Input Monitoring permission (System Settings > Privacy &
    Security > Input Monitoring) to whatever process is running Python,
    the first time it runs.
    """

    def __init__(self):
        self._present = False
        self._lock = threading.Lock()
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def _on_press(self, key):
        try:
            char = key.char
        except AttributeError:
            return

        if char == "t":
            with self._lock:
                self._present = True
            logger.info("[sensor] talk (t): presence -> True")
        elif char == "q":
            with self._lock:
                self._present = False
            logger.info("[sensor] quit (q): presence -> False")
        elif char == "r":
            logger.info("[sensor] recognize (r): noticed, not yet acted on")

    def is_present(self) -> bool:
        with self._lock:
            return self._present
