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

## Running without installing anything

For a booth machine with no Python, git, or dev tools at all: `web/`
is a self-contained static page that talks to the same public ElevenLabs
Agent directly from the browser (no backend, no build step). It's deployed
at:

https://web-theta-livid-g1pkadbf7w.vercel.app

Open that URL in any modern browser and keep the tab focused -- the USB
presence sensor's keystrokes only reach whichever tab has focus. To
redeploy after editing `web/index.html`, run `vercel --prod --yes` from
inside `web/` (linked to the `jaydenb112s-projects` Vercel scope).

By default the page uses the browser/OS's default mic and speaker. To pin
specific devices on a given machine (recommended whenever that machine has
more than one mic or speaker), append `?mic=<name>&speaker=<name>` to the
URL, e.g. `?mic=Logi&speaker=External` -- same case-insensitive substring
matching as `MIC_DEVICE_NAME`/`SPEAKER_DEVICE_NAME` in `.env`. This is
per-machine config carried in the link itself, not in the deployed code,
since one static page serves every booth machine. Give each machine the
exact link with its own params; a mismatched or unmatched name logs a
warning to the browser console and falls back to the OS default rather
than failing.

The browser page has no local echo cancellation of its own -- it relies on
the browser's built-in AEC (`getUserMedia({ audio: { echoCancellation:
true } })`) instead of `aec.py`, and there's no process supervisor, so a
hung tab needs a manual reload instead of auto-restarting.
