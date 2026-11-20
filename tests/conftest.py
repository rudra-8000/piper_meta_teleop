"""Tests must never write into the real logs/ (session.log, calibration.json) - point them at a temp dir."""
import os
import tempfile

os.environ.setdefault("PIPER_TELEOP_LOG_DIR", tempfile.mkdtemp(prefix="piper_test_logs_"))
