# Audio_AI_Experiment

Run a single presence-triggered conversation:

```bash
.venv/bin/python step8_presence.py
```

`step8_presence.py` exits after the conversation ends. For unattended
convention operation, use the asynchronous supervisor. It starts a clean
child process for every visitor and restarts failures with exponential
backoff:

```bash
.venv/bin/python convention_runner.py
```

Keep the supervisor itself under the operating system's service manager
(launchd on macOS or systemd on Linux) if it must survive terminal closure,
logout, or a machine reboot.
