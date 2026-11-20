"""YAML config -> ArmConfig + backends."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import yaml

from .arm import ArmBackend, PiperArm, SimArm
from .controller import ArmConfig

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "teleop.yaml"


def load(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_PATH) as f:
        return yaml.safe_load(f) or {}


def can_for(raw: dict, arm: str) -> str:
    """The CAN interface of a physical arm.  ``arm`` is 'left' / 'right' (looked up in ``hardware.can``) or already a CAN name.

    The CAN names come from the USB port an adapter sits in, not from the arm it is wired to, so `can_left` need not be the
    left arm.  Everything else in the project says LEFT / RIGHT meaning the physical arm and goes through this one lookup."""
    if arm in ("left", "right"):
        return (raw.get("hardware") or {}).get("can", {}).get(arm, f"can_{arm}")
    return arm


def arm_config(name: str, raw: dict, section: dict) -> ArmConfig:
    merged = {**raw.get("defaults", {}), **section}
    fields = {f.name for f in dataclasses.fields(ArmConfig)}
    kw = {k: v for k, v in merged.items() if k in fields}
    kw["name"] = name
    if "workspace" in kw:        # a per-arm workspace overrides only the axes it names
        kw["workspace"] = {**raw.get("defaults", {}).get("workspace", {}), **section.get("workspace", {})}
    for key in ("tcp_offset", "home_deg", "park_deg"):
        if key in kw:
            kw[key] = tuple(kw[key])
    return ArmConfig(**kw)


def make_backend(section: dict, raw: dict, force_sim: bool) -> ArmBackend:
    can = can_for(raw, section["can"]) if "can" in section else None
    if force_sim or section.get("backend", "piper") == "sim":
        cfg_home = np.radians(section.get("park_deg", raw.get("defaults", {}).get("park_deg", [0] * 6)))
        return SimArm(name=can or "sim", q0=cfg_home)
    p = raw.get("piper", {})
    if p.get("driver", "pyagxarm") == "pyagxarm":          # AgileX's current library (default)
        from .agx_arm import AgxPiperArm
        return AgxPiperArm(can, firmware=p.get("firmware", "auto"), speed_pct=p.get("speed_pct", 100),
                           follow_mode=p.get("follow_mode", "standard"), gripper_effort=p.get("gripper_effort", 1000))
    return PiperArm(can, dh_is_offset=p.get("dh_is_offset", 1),
                    speed_pct=p.get("speed_pct", 100), follow_mode=p.get("follow_mode", "standard"),
                    gripper_effort=p.get("gripper_effort", 1000), judge_flag=p.get("judge_flag", True))
