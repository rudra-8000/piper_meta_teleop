# Support request: PiPER gripper only adopts position targets while the command keeps changing

**To:** support@agilex.ai   **Product:** PiPER 6-DOF arm + two-finger gripper (two arms, same behaviour on both)
**Arm firmware (both arms):** S-V1.8-9   **Interface:** CAN, 1 Mbit/s, USB-CAN module, Ubuntu 22.04

## Intermittent: it moved exactly once
The gripper has moved on command only **once** in a long series of tests: on the left arm, at effort 5000 (0x159 width
10000 um, effort 5000, status 0x01), it opened 0 -> 9.1 mm (confirmed by eye) and then ignored every later target
(0-40 mm, effort 1000-30000, clear-error 0x03, disable/enable, parameter 0x47D writes before/after, ArmRobotUA's exact
Send sequence). The same sequence after a power cycle (`max_range_config` write, low-effort command, effort-5000 command)
did **not** move it again. During earlier teleoperation the gripper also moved slightly and irregularly a few times.
The position feedback is trustworthy (reads 0.0 mm when closed by hand while unpowered; 9.1-9.4 mm after the one move).

The frames used match your ArmRobotUA tool byte for byte (we checked its 0x159 / 0x47D encoders) and your pyAgxArm library.

## What was tried (all on the real arms)
1. **AgileX's own library**, `pyAgxArm` 1.0.0 (commit d90e234ba8e4..., installed from GitHub), with the documented example
   (`disable_gripper()`, `calibrate_gripper()`, `move_gripper_m(0.0 / 0.02 / 0.0, force=1.0)`), run with both `PiperFW.V189` and `PiperFW.DEFAULT`:
   - `calibrate_gripper()` returns **True** (arm ACK received); `disable_gripper()` returns **False**.
   - width stays **0.0 mm** after `move_gripper_m(0.02)`; force 0.001 - 0.009 N.
2. `piper_sdk` 0.7.0a1 (V3 interface): the official `piper_set_gripper_zero.py` sequence and variants. The arm
   **ACKs every zero** (0x476, instruction 0x75, is_set_zero_successfully = 1). The `homing_status` bit of the gripper feedback nevertheless stays False.
3. Frames verified byte-for-byte against the documented 0x159 layout (int32 width in um, int16 force in mN, status code, set-zero byte 0xAE) and identical to what `pyAgxArm.move_gripper_m` sends.
4. Status codes 0x00-0x03, the clear-error/enable handshake (0x02 then 0x01), the full official demo order (EnablePiper, 0x151 mode frames, then the gripper), send rates 50 / 200 / 500 Hz,
   arm speed ratio 30 / 50 / 100 %, arm joint motors both **enabled and disabled**, after a power cycle with the gripper closed.
5. Stroke setting `max_range_config` was 100 (factory) and was changed to 70 (0-70 mm gripper); no change in behaviour.

6. **Angle mode** (status codes 0x06 then 0x05, targets 15 / 30 / 45 deg at effort 1000 / 5000, as in
   `V3_piper_ctrl_gripper_angle.py`): the gripper obeyed disable/enable (driver_enable_status 0 -> 1) but its reported
   mode **stayed WIDTH** and the fingers did not move.
7. **Verified by camera**: a depth camera watching the gripper recorded no finger motion during width-mode commands
   (0 -> 20 -> 0 mm, effort 5000) or the angle-mode test; the change in the finger regions stayed at camera-noise level.
   The gripper's own position feedback therefore reports correctly.

8. **From-scratch controller** (bare python-can, no AgileX library): 0x159 frames only, 0x2A8 decoded directly -
   same result (status 0x40 = enabled, no faults, width mode; 0.00 mm for targets 0/20/40 mm at effort 5000; camera: no motion).
9. **Passive bus scan** of the left arm (5 s): joints 1-6 only on 0x251-0x256 / 0x261-0x266 (no 7th node), status/pose/
   joints 0x2A1-0x2A7, gripper feedback 0x2A8. Also present: **0x1C0-0x1C3** (pyAgxArm identifies these as *Revo2 hand*
   feedback; all zeros, 20-40 Hz - no hand is fitted) and **0x212** (1 Hz, `04 01 0f 02`, not in either SDK). Is the
   controller configured for a different end-effector? Is 0x212 the gripper's firmware version (4.1.15.2)?

10. **Reproduced once per power cycle** (right arm powered off, left arm power-cycled): after 1.5 s of "enable + hold
    at current opening" (0x01, effort 1000), a 10 mm target moved the gripper -0.5 -> 9.30 mm (seen by eye and camera,
    effort feedback ~160 while holding 0.7 mm short of target). Afterwards it ignored every position target in both
    directions (30/20/40/0/15 mm, effort 1000-5000, clear-error 0x03, disable/enable, hold-at-current), although it
    still obeyed enable/disable and the angle/width mode switch (mode 0 -> 1 -> 0). The earlier one-off move also stopped
    at 9.1 mm for a 10 mm target. Is there a stall / "target not reached" protection that latches until power-off?
    CAN logs (candump -L format) for this run are attached: left_battery.log, left_h1.log.

11. **Depends on the arm controller's mode.** Every move we have seen happened with the arm controller in CAN control
    mode (0x2A1 byte 0 = 0x01). On a fresh boot in STANDBY (gripper disabled at boot) nothing moved, not even the first
    target. Switching that boot to CAN mode afterwards (joint motors disabled) did not help - the gripper had already been
    enabled once in standby on that boot.
12. **Timeline.** The grippers DID follow the trigger during our first teleoperation sessions (full stream every 5 ms:
    CAN mode, joint motors enabled, 0x151 + joint targets + 0x159), unreliably: they reached 64.5 / 39.7 mm, and after
    our first set-zero attempts teleop still moved them to 27.3 / 26.0 mm. Standalone gripper commands (0x159 only, or
    with the joints disabled / arm in standby) essentially never worked. Since then we also issued many more set-zero
    variants (including 0xAE while the gripper was enabled) and changed max_range_config 100 -> 70 -> 100.
    Questions: does the gripper need the arm to be in CAN mode with joints enabled? Is there a stall / target-not-reached
    latch? Can repeated set-zero commands damage the calibration, and how is the factory calibration restored?

13. **Smooth vs stepped targets (real teleop stream, camera-verified).** A continuously varying target (sweeps,
    ramps, analog trigger) is followed over the full 0-70 mm range with ~0.2 s lag. Step targets (square waves, random
    jumps, 0<->70 pulses) mostly freeze the gripper. Rate-limiting the target to 150 mm/s made jumpy inputs move it far
    more (e.g. a 5<->65 mm square wave: 2.7 mm of motion without, 51 mm with), but a repeat run with the same limiter
    froze in segments that had worked before - the behaviour is NOT deterministic. Frame-to-frame alternation of the
    target (1 um - 1 mm) never woke it.
14. **When stuck it is not pushing.** While frozen far from its target, effort feedback is ~0.16 N*m (median; 95th pct
    0.20) - well below the 1.0 N*m limit we command and the same as while moving. A mechanically blocked finger with a
    working position loop would push at the limit; this looks like the gripper controller not adopting new targets.
    Plots and video: docs/media/ (plot_1 ... plot_5, gripper_sweeps_and_jumps.mp4, gripper_teleop_controller.mp4).

15. **Workaround found (please confirm this is expected behaviour or a firmware bug).** Adding a slow sine to the
    commanded width makes the gripper adopt targets reliably: held step targets 2/9 reached without it, 9/9 with
    +-0.25 mm @ 3 Hz, and 9/9 with +-0.5 mm at 1, 3 or 6 Hz (mean error 0.4 mm, no measurable finger jiggle). Fast
    frame-to-frame alternation (100 Hz, up to +-1 mm) does not help. With a 150 mm/s ramp + 0.5 mm @ 3 Hz sine our teleop
    holds targets within ~0.5-1 mm over the full stroke. Best setting found: no ramp + 0.5 mm @ 6 Hz (27/27 holds over 3
    runs, ~0.25-0.3 s to arrive); 10-15 Hz sines were unreliable, 25 Hz and up fail. It looks like the gripper's position loop stops adopting a
    target that is constant; is there a firmware fix, or a documented way to command it?

16. **Structured frequency dependence of the workaround.** With +-0.5 mm sine, no ramp, step targets held 1.5 s:
    2, 3 and 6 Hz reached 54/54 steps; 4, 8 and 12 Hz failed the SAME steps at the SAME positions in both passes
    (10/18, 10/18, 9/17); 10 Hz reached 2/18. This looks like an interaction with the rate at which the gripper (or the
    arm controller relaying 0x159) samples the command - what is that rate?
17. **Tracking performance with the workaround.** Policy-style target trajectories updated at 30 / 50 / 100 / 200 Hz all
    track the same: ~0.13 s lag, RMS error after lag 1.7-2.6 mm, brief pauses (>5 mm error 1-2 % of the time). Steps:
    dead time ~0.05-0.09 s, within 3 mm ~0.21-0.24 s, within 1 mm ~0.27-0.33 s (median, 10-70 mm steps).

## Current readings (read-only)
```
arm firmware         : S-V1.8-9
joint motors enabled : [False x6]
gripper              : opening 0.0 mm, driver enabled=True, homing bit=False, fault=False
gripper parameters   : travel coefficient 100, stroke setting 70, friction 1
```

## Questions
0. Could repeated `set zero` (0xAE) or changing `max_range_config` 100 -> 70 have put the gripper into a state in which it holds position but ignores targets? How is that recovered?
1. Does the gripper need a procedure or configuration beyond `calibrate_gripper()` on firmware S-V1.8-9 (for example a stroke/homing routine, or gripper firmware that must match the arm firmware)?
2. What is the `homing_status` bit of the gripper feedback meant to show, and what value should a working gripper report?
3. Could the gripper be one of the **teach-pendant / leader** type (which is a passive, force-feedback input device) rather than the follower gripper? The leader device holds torque but does not follow position targets.
   `max_range_config` read 100 from the factory, which the SDK documents as the large-gripper / pendant value.
4. If it is a hardware fault, what is the return procedure?

Please tell us the model/part number printed on the gripper if you need it; the fingers move freely by hand when the gripper is disabled.
