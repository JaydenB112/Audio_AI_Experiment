"""
Audio device lookup by name instead of raw index.

PyAudio device indices are not stable identifiers -- they're just positions
in whatever's currently connected, and shift whenever a USB device is
plugged or unplugged. Matching by name is far more resilient: it survives
reconnects, reboots, and other devices being added or removed, as long as
the device's own name string doesn't change.
"""

import pyaudio


def find_device_index(name_substring: str, *, want_input: bool) -> int:
    """Find a PyAudio device index by case-insensitive substring match on
    its name, restricted to devices that actually have the right kind of
    channel (input or output) for the requested role.

    Restricting by channel role, not just name, matters for USB devices
    that show up as two entries under the identical name (e.g. a headset's
    mic and its own speaker) -- the channel filter is what tells them apart.

    Raises RuntimeError if zero or more than one device matches, rather
    than silently guessing which one you meant.
    """
    pa = pyaudio.PyAudio()
    try:
        matches = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = info["name"]
            channels = info["maxInputChannels"] if want_input else info["maxOutputChannels"]
            if name_substring.lower() in name.lower() and channels > 0:
                matches.append((i, name))
    finally:
        pa.terminate()

    role = "input" if want_input else "output"

    if not matches:
        raise RuntimeError(
            f"No {role} device found matching {name_substring!r}. "
            "Run list_audio_devices.py to see what's currently connected."
        )
    if len(matches) > 1:
        found = ", ".join(f"[{i}] {name!r}" for i, name in matches)
        raise RuntimeError(
            f"Multiple {role} devices match {name_substring!r}: {found}. "
            "Use a more specific substring to pick one."
        )

    return matches[0][0]
