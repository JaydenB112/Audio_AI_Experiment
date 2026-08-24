# Audio_AI_Experiment

Run a single presence-triggered conversation, using the local Deepgram/
Claude/ElevenLabs-TTS pipeline (Pipecat-orchestrated STT, LLM, and TTS):

```bash
.venv/bin/python step8_presence.py
```

Or using the ElevenLabs Conversational AI Agents platform instead (STT,
LLM, and TTS all run on ElevenLabs' side; conversation history is in the
ElevenLabs dashboard, not local). Requires an Agent already configured in
the ElevenLabs dashboard -- see `.env.example` for `ELEVENLABS_AGENT_ID`:

```bash
.venv/bin/python step9_elevenlabs_agent.py
```

Both exit after the conversation ends. For unattended convention
operation, use the asynchronous supervisor. It starts a clean child
process for every visitor and restarts failures with exponential backoff
(wired to step9_elevenlabs_agent.py):

```bash
.venv/bin/python convention_runner.py
```

Keep the supervisor itself under the operating system's service manager
(launchd on macOS or systemd on Linux) if it must survive terminal closure,
logout, or a machine reboot.
