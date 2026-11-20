#!/usr/bin/env python3
"""Check, zero, test and configure the grippers / teach pendant.

    python tools/gripper_setup.py status can_left can_right          # read-only: opening, enabled, ZEROED, faults
    python tools/gripper_setup.py zero   can_left can_right          # guided zeroing - YOU close each gripper by hand
    python tools/gripper_setup.py test   can_left can_right          # tiny 3 mm open/close to prove it follows commands
    python tools/gripper_setup.py diagnose can_left can_right        # read-only bundle to paste into a support request
    python tools/gripper_setup.py params can_left --range 70         # stroke / pendant travel settings
    python tools/gripper_setup.py read   can_left                    # pendant parameter feedback

status : the gripper's own report.  NOTE: the "homing bit" it shows is NOT a reliable "zeroed" indicator (on the first real
         arms it stayed False although the arm ACKnowledged every zero).  The real tests are  zero  (arm ACK) and  test.
zero   : AgileX's own gripper-setup procedure starts with "disable the arm", so this first DISABLES THE ARM'S JOINT MOTORS (only if the arm is in
         its folded rest pose; the wrist then sags ~25 degrees, as when unpowered), then releases the gripper, has you
         close it by hand (skipped if it already reads closed), sends the zero until the ARM ACKNOWLEDGES it, and checks
         the gripper really follows a command.  The arm is left disabled; the teleop server re-enables it.
           --disable-arm   consent to the arm disable without the interactive question
           --stroke 70     also set the stroke setting to 70 mm (AgileX's own V2_piper_set_gripper_param.py uses 100/70/1)
test   : opens 10 mm and back through the gripper input layer (gripper only; no arm joint moves).
params : GripperTeachingPendantParamConfig(teaching_range_per, max_range_config, teaching_friction)
           --scale    pendant travel coefficient 100..200 (%)
           --range    max stroke in mm: 70 (small gripper / pendant), 100 (large), 0 = invalid
           --friction pendant friction 1..10
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piper_teleop import gripper as G  # noqa: E402
from piper_teleop.arm import PiperArm, _make_interface, is_folded  # noqa: E402

ACTIONS = ("status", "zero", "test", "params", "read", "diagnose")


def connect(can):
    p = _make_interface(can, 1, True)
    p.ConnectPort()
    time.sleep(0.5)
    return p


def show(can, p):
    st = G.status(p)
    if st is None:
        print(f"  {can}: gripper status unreadable")
        return None
    print(f"  {can}: opening {st['value_m'] * 1000:6.1f} mm | effort {st['effort']:5d} | enabled: {st['enabled']} | "
          f"homing bit (unreliable): {st['zeroed']} | fault: {st['error']}")
    return st


def stroke(can, p, want):
    p.ArmParamEnquiryAndConfig(4)
    time.sleep(0.5)
    fb = p.GetGripperTeachingPendantParamFeedback().arm_gripper_teaching_param_feedback
    print(f"  stroke setting: max_range_config = {fb.max_range_config}  (70 = small 0-70 mm gripper, 100 = large)")
    if fb.max_range_config == 70:
        return
    if want is None:
        want = input("  Your manual lists these grippers as 0-70 mm. Set the stroke setting to 70 now? [y/N] ").strip().lower() == "y"
    if want:
        p.GripperTeachingPendantParamConfig(fb.teaching_range_per or 100, 70, fb.teaching_friction or 1)
        time.sleep(0.4)
        print("  stroke set to 70 mm.")


def prepare_arm(can, p, consent):
    """The gripper only accepted zero/stroke writes with the arm's joint motors disabled."""
    en = G.arm_motors_enabled(p)
    if en is not None and not any(en):
        print("  arm joint motors already disabled.")
        return True
    q = PiperArm(can, iface=p).read().q
    if not is_folded(q):
        print("  [STOP] the arm is not in its folded rest pose, and disabling it would make it DROP. Fold it first "
              "(python tools/go_rest.py " + can + "), then run this again.")
        return False
    if not consent:
        print("  Gripper zeroing needs the arm's JOINT MOTORS DISABLED (AgileX's own procedure starts with that).")
        print("  The arm is folded at rest, so only the wrist will sag ~25 degrees, exactly as when the arm is unpowered.")
        if input("  Type YES to disable this arm's joint motors now: ").strip() != "YES":
            print("  skipped.")
            return False
    ok = G.disable_arm(p, consent=True)          # reached only after --disable-arm or the typed YES
    print("  arm joint motors disabled." if ok else "  [WARN] could not confirm the joint motors are disabled.")
    return ok


def zero(can, p, consent, stroke_want):
    print(f"\n=== {can}")
    st0 = show(can, p)
    if not prepare_arm(can, p, consent):
        return False
    stroke(can, p, stroke_want)
    print("  releasing the gripper (it goes limp)...")
    G._send(p, 1.5, 0, 0x00)
    st = G.status(p)
    if st is not None and st["value_m"] * 1000 < 1.5:
        print("  the gripper already reads fully closed - not asking you to close it again.")
    else:
        input(f"  CLOSE the fingers of the {can} gripper completely BY HAND, then press Enter... ")
    res = G.try_zero(p)
    show(can, p)
    if not res["ok"]:
        print("  [FAIL] the arm did not acknowledge the zero command.")
        print("         Next: (1) power-cycle the ARM with the gripper closed, then  status  and  test.")
        print("               (2) With the arm powered and the gripper enabled, can you still push the fingers by hand freely? "
              "If yes the gripper motor is not producing torque.")
        print("               (3) Try the gripper in AgileX's ArmRobotUA host software; if it fails there too it is not this "
              "software. Then:  python tools/gripper_setup.py diagnose " + can + "  and send the output to support@agilex.ai.")
        return False
    print(f"  [ OK ] the arm ACKNOWLEDGED the zero (variant '{res['variant']}').")
    r = G.follow_test(p)
    print(f"  follow-test: moved {r['moved_mm']} mm -> " + ("FOLLOWS commands." if r["followed"] else
          "did NOT follow. Check power/blockage; try  test  after a power-cycle."))
    return r["followed"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="+", help="an action and one or more CAN names (either order)")
    ap.add_argument("--range", type=int, default=70, choices=[0, 70, 100])
    ap.add_argument("--scale", type=int, default=100)
    ap.add_argument("--friction", type=int, default=1)
    ap.add_argument("--disable-arm", action="store_true", help="consent to disabling the (folded) arm's joint motors for zeroing")
    ap.add_argument("--stroke", type=int, choices=[70], help="also set the stroke setting to 70 mm without asking")
    a = ap.parse_args()
    actions = [i for i in a.items if i in ACTIONS]
    cans = [i for i in a.items if i not in ACTIONS]
    if len(actions) != 1 or not cans:
        ap.error("give exactly one action (status|zero|test|params|read) and at least one CAN name")
    act, ok = actions[0], True
    for can in cans:
        p = connect(can)
        if act == "status":
            show(can, p)
        elif act == "zero":
            ok &= zero(can, p, a.disable_arm, True if a.stroke == 70 else None)
        elif act == "diagnose":
            print(f"=== {can}")
            print("\n".join("  " + l for l in G.diagnose(p)))
        elif act == "test":
            show(can, p)
            input(f"  {can}: the gripper will move 10 mm and return (no arm joint moves). Enter to go... ")
            r = G.follow_test(p)
            print(f"  moved {r['moved_mm']} mm -> " + ("FOLLOWS commands" if r["followed"] else "does NOT follow commands"))
            ok &= r["followed"]
        else:
            if act == "params":
                p.GripperTeachingPendantParamConfig(a.scale, a.range, a.friction)
                time.sleep(0.1)
            p.ArmParamEnquiryAndConfig(4)
            time.sleep(0.3)
            print(f"{can}: {p.GetGripperTeachingPendantParamFeedback()}")
            show(can, p)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
