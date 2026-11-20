"""Where logs go.  Tests point PIPER_TELEOP_LOG_DIR at a temp dir so they never touch the real session log."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def log_dir() -> Path:
    d = Path(os.environ.get("PIPER_TELEOP_LOG_DIR") or ROOT / "logs")
    d.mkdir(parents=True, exist_ok=True)
    return d
