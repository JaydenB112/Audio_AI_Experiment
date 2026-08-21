"""
Diagnostic tool, not a pipeline step.

Lists every audio device your system can see, with its PyAudio index, name,
max input/output channels, and default sample rate. Pair a Bluetooth device
in your OS's Bluetooth settings first, then run this to find its index and
see what sample rate it actually reports.

Run:
    python list_audio_devices.py
"""

import pyaudio

pa = pyaudio.PyAudio()

for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    print(
        f"[{i}] {info['name']!r} "
        f"in={int(info['maxInputChannels'])} "
        f"out={int(info['maxOutputChannels'])} "
        f"default_sample_rate={int(info['defaultSampleRate'])}"
    )

pa.terminate()
