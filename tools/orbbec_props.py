#!/usr/bin/env python3
"""List the colour-sensor properties an Orbbec camera exposes (current value + range), read-only.

    PYTHONPATH=$HOME/orbbec_v1_legacy/install/lib .venv_cam/bin/python tools/orbbec_props.py [--serial S]
"""
import argparse

from pyorbbecsdk import Context, OBLogLevel, OBPermissionType, OBPropertyID

ap = argparse.ArgumentParser()
ap.add_argument("--serial")
a = ap.parse_args()
try:
    Context.set_logger_level(OBLogLevel.NONE)
except Exception:
    pass
dl = Context().query_devices()
for i in range(dl.get_count()):
    dev = dl.get_device_by_index(i)
    info = dev.get_device_info()
    if a.serial and info.get_serial_number() != a.serial:
        continue
    print(f"== {info.get_name()} {info.get_serial_number()}  fw {info.get_firmware_version()}")
    for name in dir(OBPropertyID):
        if not name.startswith("OB_PROP_") or "COLOR" not in name:
            continue
        pid = getattr(OBPropertyID, name)
        try:
            if not dev.is_property_supported(pid, OBPermissionType.PERMISSION_READ):
                continue
        except Exception:
            continue
        row = None
        for kind, getter, rng in (("bool", "get_bool_property", "get_bool_property_range"), ("int", "get_int_property", "get_int_property_range"),
                                  ("float", "get_float_property", "get_float_property_range")):
            if not name.endswith(("_BOOL", "_INT", "_FLOAT")) or not name.endswith(kind.upper()):
                continue
            try:
                cur = getattr(dev, getter)(pid)
                r = getattr(dev, rng)(pid)
                row = f"{name:48s} {kind:5s} now {cur!s:8s} range [{r.min}..{r.max}] step {r.step} default {r.default}"
            except Exception as e:
                row = f"{name:48s} {kind:5s} (read failed: {str(e)[:40]})"
        if row:
            print("  " + row)
