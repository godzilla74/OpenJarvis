"""``jarvis voice`` command group — train, setup, and start the voice assistant."""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

from openjarvis.core.config import load_config

console = Console()

_MODEL_PATH = Path.home() / ".openjarvis" / "models" / "hey_jarvis.onnx"


@click.group("voice")
def voice() -> None:
    """Always-on voice assistant — say 'Hey Jarvis' to start."""


@voice.command("train")
@click.option("--samples", default=20, show_default=True,
              help="Number of TTS samples per voice (more = better accuracy, longer training)")
def voice_train(samples: int) -> None:
    """Generate synthetic training data and train the 'Hey Jarvis' wake word model.

    Requires macOS (uses the built-in `say` command). Takes ~10-20 minutes.
    Only needs to be run once.
    """
    from openjarvis.voice.train_wake_word import run_training

    if _MODEL_PATH.exists():
        click.confirm(
            f"Model already exists at {_MODEL_PATH}. Re-train?",
            abort=True,
        )

    console.print("[bold]Training 'Hey Jarvis' wake word model...[/bold]")
    model_path = run_training(n_per_voice=samples)
    console.print(f"[green]Done! Model saved to {model_path}[/green]")
    console.print("Run [bold]jarvis voice setup[/bold] next to connect your accounts.")


@voice.command("setup")
def voice_setup() -> None:
    """Connect Gmail, Calendar, and Tasks accounts via OAuth.

    Run this once before starting the voice assistant for the first time.
    """
    console.print("[bold]Voice Assistant Setup[/bold]\n")
    _setup_gmail()
    _setup_calendar()
    _setup_tasks()
    console.print("\n[green]Setup complete! Run [bold]jarvis voice start[/bold] to begin.[/green]")


def _setup_gmail() -> None:
    console.print("[bold]Gmail accounts[/bold]")
    console.print("You can connect multiple Gmail accounts. Press Enter with no input when done.\n")
    accounts = []
    idx = 1
    while True:
        label = click.prompt(f"Account {idx} label (e.g. 'work email', 'personal')", default="")
        if not label:
            break
        console.print(f"Opening browser for OAuth — sign in to your {label} Gmail account...")
        token = _run_google_oauth(
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/gmail.send",
            ],
            label=f"gmail_{label.replace(' ', '_')}",
        )
        if token:
            accounts.append({"label": label, "token_path": token})
            console.print(f"[green]  ✓ {label} connected[/green]")
        idx += 1
    console.print(f"Connected {len(accounts)} Gmail account(s).")


def _setup_calendar() -> None:
    console.print("\n[bold]Google Calendar[/bold]")
    console.print("Opening browser for OAuth...")
    _run_google_oauth(
        scopes=["https://www.googleapis.com/auth/calendar"],
        label="gcalendar",
    )
    console.print("[green]  ✓ Calendar connected[/green]")


def _setup_tasks() -> None:
    console.print("\n[bold]Google Tasks[/bold]")
    console.print("Opening browser for OAuth...")
    _run_google_oauth(
        scopes=["https://www.googleapis.com/auth/tasks"],
        label="gtasks",
    )
    console.print("[green]  ✓ Tasks connected[/green]")


def _run_google_oauth(*, scopes: list[str], label: str) -> str | None:
    """Run the OAuth flow and save the token. Returns the token file path."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import]
        from openjarvis.core.config import DEFAULT_CONFIG_DIR

        client_secrets = DEFAULT_CONFIG_DIR / "google_client_secret.json"
        if not client_secrets.exists():
            console.print(
                f"[red]Missing {client_secrets}[/red]\n"
                "Download your OAuth 2.0 client credentials from Google Cloud Console\n"
                "(APIs & Services → Credentials → Download JSON) and save to that path."
            )
            return None

        flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), scopes=scopes)
        creds = flow.run_local_server(port=0)
        token_path = DEFAULT_CONFIG_DIR / f"{label}_token.json"
        with open(token_path, "w") as f:
            f.write(creds.to_json())
        return str(token_path)
    except Exception as exc:
        console.print(f"[red]OAuth failed: {exc}[/red]")
        return None


@voice.command("start")
@click.option("--device", default="", help="Microphone device name fragment (e.g. 'Logitech')")
@click.option("--stt-model", default="large-v3", show_default=True, help="Faster-Whisper model size")
@click.option("--local-model", default="qwen2.5:14b", show_default=True, help="Local Ollama model")
def voice_start(device: str, stt_model: str, local_model: str) -> None:
    """Start the always-on voice assistant."""
    if not _MODEL_PATH.exists():
        console.print(
            "[red]Wake word model not found.[/red] "
            "Run [bold]jarvis voice train[/bold] first."
        )
        sys.exit(1)

    from openjarvis.voice.capture import find_device_index
    from openjarvis.engine.ollama import OllamaEngine

    mic_device_idx = None
    if device:
        mic_device_idx = find_device_index(device)
        if mic_device_idx is None:
            console.print(f"[yellow]Warning: no device matching '{device}' found, using default[/yellow]")

    cfg = load_config()
    tokens = _load_tokens(cfg)

    local_engine = OllamaEngine()
    cloud_engine = _get_cloud_engine(cfg)

    from openjarvis.voice.loop import VoiceLoop

    loop = VoiceLoop(
        wake_word_model=_MODEL_PATH,
        stt_model=stt_model,
        mic_device=mic_device_idx,
        local_engine=local_engine,
        local_model=local_model,
        cloud_engine=cloud_engine,
        cloud_model=cfg.voice_assistant.cloud_model,
        gmail_tokens=tokens["gmail"],
        calendar_token=tokens.get("gcalendar", ""),
        tasks_token=tokens.get("gtasks", ""),
    )
    loop.run()


def _load_tokens(cfg) -> dict:
    """Load OAuth tokens from disk. Returns {service: token_string} dict."""
    import json
    from openjarvis.core.config import DEFAULT_CONFIG_DIR

    tokens: dict = {"gmail": {}}

    for token_file in DEFAULT_CONFIG_DIR.glob("gmail_*_token.json"):
        label = token_file.stem.replace("gmail_", "").replace("_token", "")
        try:
            data = json.loads(token_file.read_text())
            tokens["gmail"][label] = data.get("token", "")
        except Exception:
            pass

    for service in ("gcalendar", "gtasks"):
        token_file = DEFAULT_CONFIG_DIR / f"{service}_token.json"
        if token_file.exists():
            try:
                data = json.loads(token_file.read_text())
                tokens[service] = data.get("token", "")
            except Exception:
                pass

    return tokens


def _get_cloud_engine(cfg):
    """Return the configured cloud engine (Anthropic Claude via CloudEngine)."""
    try:
        from openjarvis.engine.cloud import CloudEngine  # FIX: was AnthropicEngine
        return CloudEngine()
    except Exception:
        from openjarvis.engine.ollama import OllamaEngine
        return OllamaEngine()
