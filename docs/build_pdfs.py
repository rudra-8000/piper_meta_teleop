#!/usr/bin/env python3
"""Rebuild both PDF guides:  .venv/bin/python docs/build_pdfs.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import setup_guide, teleop_guide  # noqa: E402

setup_guide.build(str(HERE / "Piper_Setup_Guide.pdf"))
teleop_guide.build(str(HERE / "Teleop_User_Guide.pdf"))
print("built", *(p.name for p in sorted(HERE.glob("*.pdf"))))
