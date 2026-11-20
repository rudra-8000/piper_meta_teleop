#!/usr/bin/env python3
"""Run AgileX's OWN gripper example (pyAgxArm) on one arm, to find out whether THEIR library makes the gripper move.

NOT part of the teleop software and NOT tested by the author (pyAgxArm could not be installed in the authoring
environment).  It follows the documented API (docs/effector/agx_gripper/agx_gripper_api.md in
github.com/agilexrobotics/pyAgxArm).  Run it in a THROWAWAY virtual environment of your own:

    python3 -m venv ~/agx_check && ~/agx_check/bin/pip install python-can git+https://github.com/agilexrobotics/pyAgxArm.git
    ~/agx_check/bin/python tools/agx_gripper_check.py can_left            # firmware S-V1.8-9  -> PiperFW.V189
    ~/agx_check/bin/python tools/agx_gripper_check.py can_left --fw default

What it does (gripper only; no arm joint is commanded):  reads the gripper status, disables the gripper, calibrates
(zero) it with the gripper closed, then asks it to open 2 cm and close again, printing what the gripper reports.
Stop the teleop server first - only one program should drive an arm at a time.
"""
import argparse
import time

from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

ap = argparse.ArgumentParser()
ap.add_argument("can")
ap.add_argument("--fw", choices=["v189", "default"], default="v189")
a = ap.parse_args()

cfg = create_agx_arm_config(robot=ArmModel.PIPER, firmeware_version=PiperFW.V189 if a.fw == "v189" else PiperFW.DEFAULT,
                            channel=a.can)
robot = AgxArmFactory.create_arm(cfg)
eff = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
robot.connect()
time.sleep(0.8)


def show(label):
    gs = eff.get_gripper_status()
    if gs is None:
        print(f"  {label}: no gripper feedback")
        return
    m = gs.msg
    print(f"  {label}: width {m.value * 1000:6.1f} mm | force {m.force:5.3f} N | enabled {m.foc_status.driver_enable_status}")


print("gripper communication ok:", eff.is_ok())
show("start")
robot.set_speed_percent(100)
print("disable_gripper():", eff.disable_gripper())
input("Close the gripper completely by hand, then press Enter... ")
print("calibrate_gripper() (arm ACK):", eff.calibrate_gripper())
for width in (0.0, 0.02, 0.0):
    eff.move_gripper_m(value=width, force=1.0)
    time.sleep(2.0)
    show(f"after move_gripper_m({width})")
print("If the width above never left 0 mm, AgileX's own library does not move this gripper either: send the output "
      "to support@agilex.ai together with  tools/gripper_setup.py diagnose  output.")
