from pdfkit import *  # noqa: F401,F403


def build(path):
    doc = Doc(path, "AgileX Piper - Unboxing and Computer Setup Guide", "Piper setup guide - unboxing to first light")
    s = Story()
    s += [SP(30), P("Piper setup guide", "title"),
          P("From the unopened box to two arms talking to your Ubuntu PC. No ROS - plain Python and CAN.", "sub")]
    s += callout("danger", "Read this before you plug anything in",
                 "A Piper is a 4.2 kg arm that can move at up to 180-225 deg/s. It has no collision sensing. "
                 "Keep your head and hands out of its reach while it is powered, never wear loose sleeves or hang "
                 "jewellery near it, and keep a hand next to the power switch / plug for the first runs. "
                 "AgileX's manual states the arm has no built-in obstacle avoidance and that you should operate it "
                 "within sight in an open area.")
    s += table([["What", "Where it is covered"],
                ["1. What is in the box, what you still need", "Chapter 1"],
                ["2. How everything connects (power, CAN, USB)", "Chapter 2"],
                ["3. Mounting two arms", "Chapter 3"],
                ["4. First power-up, the zero pose, the teach button", "Chapter 4"],
                ["5. Computer setup: software, USB-CAN adapters, naming them", "Chapter 5"],
                ["6. First-contact tests (read-only, then enable, then a 3-degree nudge)", "Chapter 6"],
                ["7. Gripper and teach pendant", "Chapter 7"],
                ["8. Leader / follower roles, firmware, cameras, shutdown, troubleshooting", "Chapters 8-12"]],
               [120, 50])
    s += callout("note", "How this guide was made - and its limits",
                 "Hardware facts come from AgileX's PiPER Quick Start User Manual V1.0 (2024.09) and the official "
                 "<font name='Mono'>piper_sdk</font> documentation; links are at the end. The software in this "
                 "project was tested against a simulator, unit tests and the SDK's own kinematics, but <b>not</b> on "
                 "real arms (they were still boxed when it was written). Where the manual is silent - for example "
                 "the colours of the base status light - this guide says so instead of guessing.")
    s.append(PageBreak())

    # ------------------------------------------------------------------ 1
    s += [H1("1. What is in the box"), P("Per the quick-start manual, each arm ships with:")]
    s += table([["Item", "Qty", "Notes"],
                ["6-DOF Piper arm", "1", "4.2 kg, 1.5 kg payload, 626.75 mm working radius, repeatability +-0.1 mm"],
                ["USB-to-CAN module", "1", "The arm's PC link. Linux sees it as a SocketCAN device (gs_usb driver)"],
                ["Power adapter", "1", "DC 24 V. See the power warning below"],
                ["MicroUSB cable", "1", "Presumably module-to-PC; the module's own markings win"],
                ["Power and communication cable", "1", "Has the aviation plug that goes into the arm's base"],
                ["Base mounting screws / wrench", "4 / 1", "M5, hole spacing 70 mm"]],
               [55, 12, 103])
    s += table([["Your optional parts", "What the manual says"],
                ["Two-finger gripper (follower)", "500 g, 0-70 mm opening, rated clamp 40 N (max 50 N), 24 V, CAN, flange mount"],
                ["Teach pendant (\"leader gripper\")", "550 g, 0-70 mm, 40 N with force control and force feedback, 24 V, CAN, flange mount. "
                 "It is a hand-held gripper-shaped input device that bolts to the <b>leader</b> arm's flange so you can squeeze it to command the follower's gripper"],
                ["Cameras and mounts", "Not part of the arm. They plug into the PC over USB (chapter 11)"]],
               [55, 115])
    s += callout("warn", "You have 2 arms: you need 2 USB-CAN modules and 2 power supplies",
                 "Both arms answer on identical CAN IDs, so they cannot share one bus unless you are using AgileX's "
                 "native master/slave linkage (where that is exactly the point). For anything driven from the PC, "
                 "each arm gets its own CAN module and its own power adapter. The two modules look identical - "
                 "chapter 5 shows how to tell them apart.")
    s += callout("danger", "Power supply limits (manual p.6)",
                 "Standard supply is DC 24 V (min 24 V, max 26 V). If you ever use a different supply: <b>never above "
                 "26 V</b>, and it must deliver <b>at least 10 A</b>. Typical draw is about 40 W, peak up to 120 W. "
                 "The gripper and pendant are also listed as 24 V devices; check how yours is fed (through the flange "
                 "connector from the arm) before assuming anything.")

    # ------------------------------------------------------------------ 2
    s += [KeepTogether([H1("2. How everything connects"), wiring_diagram()]), SP(6)]
    s += [H3("The aviation plug (arm base)"),
          P("The arm's base has an electrical panel with three items (manual figure 2.1.2): <b>1</b> power-and-communication "
            "port, <b>2</b> status indicator light, <b>3</b> J1/J2 connection port. The cable's aviation plug carries "
            "<b>4</b> power +, <b>5</b> power -, <b>6</b> CAN-H, <b>7</b> CAN-L."),
          ] + bullets([
        "Align the <b>red dot</b> on the plug with the red dot on the socket and push straight in.",
        "To remove: press down on the textured collar and pull. Do not twist or pull on the cable.",
        "Lead the CAN wires (CAN_H, CAN_L) out of the cable to the matching terminals on the USB-CAN module; plug the "
        "module into the PC. If your module is labelled differently from this text, trust the label.",
        "Triple-check power polarity (4 is +, 5 is -) before switching on. The manual does not describe reverse-polarity protection."])
    s += callout("note", "One cable, two jobs",
                 "Power and CAN travel through the same aviation plug. So 'arm has no feedback' is either a power problem, "
                 "a CAN wiring problem, or a role problem (chapter 8) - and the checks in chapter 6 tell them apart.")

    # ------------------------------------------------------------------ 3
    s += [H1("3. Mounting two arms")]
    s += bullets([
        "Bolt each base to something rigid and heavy: four M5 screws on a 70 mm x 70 mm square. A wobbly table shows up as shaky teleop.",
        "Mount with the cable side facing <b>backward</b> (the SDK's installation-position setting assumes wiring at the rear).",
        "Reach is 626.75 mm. Two layouts work with this software: <b>side by side</b> facing the same way (bases about 40-50 cm apart), or "
        "<b>ALOHA style, facing each other</b> across the table. For facing arms, spacing matters: after the start move each "
        "fingertip is about 0.22 m out from its base, so the tips end up (separation - 0.43 m) apart, and the workspace limit "
        "<font name='Mono'>workspace.x[1]</font> in <font name='Mono'>config/teleop.yaml</font> must stay below about (separation / 2 - 0.05 m): "
        "0.55 m apart gives 0.22, 0.60 m gives 0.25, 0.70 m gives 0.30, 0.80 m or more allows the 0.32 default. "
        "<b>The first dual-arm start collided because the old start pose reached 0.32 m out from each base.</b> These figures are my arithmetic, not AgileX's.",
        "Leave a clear half-metre around each arm, and keep cables away from J1 rotation. At rest the elbow folds <b>behind</b> the base (about 0.28 m), "
        "so do not put a wall or the other arm's cable run directly behind a base.",
        "The teleop software works in each arm's own base frame (x forward, y left, z up). Side by side, both facing the same way, "
        "'move the hands forward' moves both arms forward. For facing arms set <font name='Mono'>base_yaw_deg</font> per arm "
        "(-90 for the arm on your left, +90 for the arm on your right when you stand at the table's long side; swap them if it "
        "moves the wrong way) - see the user guide."])

    # ------------------------------------------------------------------ 4
    s += [H1("4. First power-up, the zero pose, the teach button")]
    s += steps([
        "Fold the arm into its <b>zero pose</b> by hand while unpowered (the 'Robotic Arm Zero Point' figure in the manual: "
        "the folded rest posture). The arm should be there whenever you switch modes - the manual says that switching from "
        "link/teach mode into CAN control requires the arm to be at the zero point.",
        "Make sure the area is clear, then connect PSU and switch on. The base has a status indicator light; the manual "
        "gives no colour table for it, so note what yours shows and compare between the two arms.",
        "Do <b>not</b> press the teach button yet (next paragraph)."])
    s += callout("danger", "The button between J5 and J6 is NOT the power button",
                 "It is the <b>teach / drag-record</b> button. Pressing it puts the arm into <b>teaching mode</b> (green light = recording), "
                 "which is not what the PC software wants. Power is the supply / its switch. "
                 "<b>Never change an arm's control mode while it is held up in the air</b>: switching out of teaching mode (or standby) from a raised "
                 "pose can make the arm drop. This happened on the first dual-arm test. Fold the arm to its rest pose first (with the power on in teach "
                 "mode you can lower it by hand while supporting it, or power it off and fold it by hand). The software now refuses to change mode "
                 "unless J2 and J3 are at their folded zero stops.")
    s += table([["Teach-button light (between J5 and J6)", "Meaning (manual 2.2)"],
                ["Off", "Drag-teach stopped. <b>Required</b> before the PC may control the arm"],
                ["Solid green", "Recording a dragged trajectory (max 3 minutes; each new recording erases the last)"],
                ["Flashing green (500 ms)", "Playing back the recorded trajectory - keep clear"]],
               [70, 100])
    s += P("Single click: start/stop recording. Double click: play back the last recording. "
           "Always end with the light <b>off</b>. To go to PC control: light off, then standby, then CAN mode "
           "(our software does the last two for you).")

    # ------------------------------------------------------------------ 5
    s.append(PageBreak())
    s += [H1("5. Computer setup (Ubuntu 22.04 / 24.04)"),
          P("One PC runs everything. (If you have two computers, use the faster one; there is nothing to split across machines.) "
            "The project lives in <font name='Mono'>piper_meta_teleop/</font>."),
          H3("5.1 Install"),
          ] + code("cd piper_meta_teleop\nscripts/install.sh") + [
        P("This installs can-utils, ethtool, iproute2, adb and openssl with apt (asks for sudo), creates "
          "<font name='Mono'>.venv</font>, installs the Python packages and runs the self-test. The AgileX SDK is installed "
          "from GitHub on purpose: the PyPI release (0.6.2) lacks the V3 interface that firmware 1.8-8 and newer needs; "
          "the GitHub version provides V3 and the code falls back to V2 for older firmware automatically."),
        H3("5.2 Plug in the CAN adapters one at a time"),
        P("The two modules are identical, so the only stable way to tell them apart is <b>the USB port they are plugged "
          "into</b>. Pick two ports, label them 'LEFT' and 'RIGHT' with tape, and always use them the same way.")] + steps([
        "Unplug both modules. Plug in the LEFT one.",
        "<font name='Mono'>scripts/can_setup.sh list</font> - it prints something like "
        "<font name='Mono'>can0   usb-port 1-2:1.0</font>. Note the port.",
        "Unplug it, plug in the RIGHT one in its own port, run <font name='Mono'>list</font> again, note that port.",
        "Plug both in. Copy <font name='Mono'>config/can_ports.conf.example</font> to <font name='Mono'>config/can_ports.conf</font> "
        "and put your two ports in it.",
        "<font name='Mono'>scripts/can_setup.sh all</font> - renames them <b>can_left</b> / <b>can_right</b>, sets 1 Mbit/s, brings them up.",
        "Optional: <font name='Mono'>scripts/can_setup.sh persist</font> writes a udev rule so the names survive re-plugging. "
        "You still run <font name='Mono'>all</font> after each boot to bring the links up."]) + [
        P("Check: <font name='Mono'>ip -br link show type can</font> must show both names with state UP (or UNKNOWN, which is normal for CAN). "
          "The bit rate is fixed at 1,000,000 for Piper; the SDK documentation says only the arm's own CAN module is supported.")]
    s += callout("note", "No adapter showing up?",
                 "<font name='Mono'>lsusb</font> should list a device; if it does but there is no canX, load the driver: "
                 "<font name='Mono'>sudo modprobe gs_usb</font>. On a stock Ubuntu kernel gs_usb ships in "
                 "<font name='Mono'>linux-modules-extra-$(uname -r)</font>. If you use a different CAN module "
                 "(serial/slcan, PCIe) the SDK needs <font name='Mono'>judge_flag: false</font> - see the SDK Q&amp;A.")
    s += [H3("5.3 Watching raw CAN traffic (optional)"),
          P("With an arm powered and connected: <font name='Mono'>candump can_left</font>. A healthy single arm sends on the order of "
            "3,000 frames/s (the SDK's own figure is about 3040 fps in <font name='Mono'>GetCanFps()</font>). "
            "Silence means power, wiring, or the arm being in leader mode.")]

    # ------------------------------------------------------------------ 6
    s += [H1("6. First-contact tests - in this order")]
    s += table([["Command", "What it does", "Moves the arm?"],
                [".venv/bin/python tools/hw_check.py can_left can_right", "CAN up at 1 Mbit/s? feedback frames? joint angles, firmware, status, gripper reading", "No"],
                ["... hw_check.py can_left --enable", "Also enables the motors (holding torque on)", "No (arm may stiffen)"],
                ["... hw_check.py can_left --enable --nudge", "Rotates joint 1 by +3 degrees slowly (20 % speed) and back; verifies the angle change", "Yes, 3 deg"],
                ["tools/go_rest.py can_left can_right --plan", "Shows each arm's pose and mode and the planned move to the folded rest pose. Without --plan it makes the move (0.3 rad/s, gripper untouched) and leaves the motors enabled and holding", "No / Yes"]],
               [62, 80, 28])
    s += P("Note on the 'rest' pose: powered and held at zero, the wrist is straight (J5 = 0); with the motors off the wrist droops (J5 about 20-30 degrees on "
           "the arms this was developed with) while J1-J4 and J6 stay within a few degrees of zero and J2/J3 sit on their zero stops. Both are the same folded rest.")
    s += P("Do the left arm completely, then the right. Only continue when every line says OK. "
           "<font name='Mono'>hw_check</font> prints the firmware version: <b>1.8-8 or newer</b> uses the V3 interface, older "
           "firmware uses V2 (handled automatically). Firmware before S-V1.6-3 needs <font name='Mono'>dh_is_offset: 0</font> "
           "in <font name='Mono'>config/teleop.yaml</font> - new arms should not.")
    s += callout("warn", "Nothing in this project cuts motor power unless you ask",
                 "A disabled Piper is not a rigid object: its wrist drops (J5 falls by 20-30 degrees from a held rest pose). So "
                 "<font name='Mono'>hw_check</font>, <font name='Mono'>go_rest</font>, the leader/follower tool and the Quest server all leave the motors "
                 "<b>enabled and holding</b> when they finish. Only the explicit flags <font name='Mono'>--release</font> / "
                 "<font name='Mono'>--release-on-exit</font> disable them, and only at the folded pose. When you are done for the day, cut the supply with the arm at rest.")

    # ------------------------------------------------------------------ 7
    s += [H1("7. Gripper and teach pendant")]
    s += bullets([
        "Both mount on the arm's flange (manual 2.4) and are controlled over the arm - you never address them separately.",
        "<b>Follower arm</b> gets the two-finger gripper. <b>Leader arm</b> gets the teach pendant (for arm-to-arm teleop, chapter 3 of the user guide). "
        "For Quest teleop, both arms get a normal gripper and the pendant is unused.",
        "Gripper commands are an opening in 0.001 mm units, 0-70 mm; effort 0-5 N.m in units of 0.001 N.m. The software starts at 1.0 N.m - "
        "keep it low until you know what you are gripping."])
    s += P("<b>Gripper status (read this carefully - an earlier version of this guide was wrong).</b> On the first real Pipers (arm firmware S-V1.8-9) both grippers are connected, "
           "enabled, report a correct opening, and <b>hold torque when enabled</b> (they resist your hand) - but they <b>ignore new position targets</b>: no motion, almost no effort, "
           "whatever target (3-50 mm), effort (0.5-2.5 N.m), send rate (50-500 Hz), enable sequence or arm speed setting is used. The 'homing bit' in the gripper feedback stays False, "
           "but it is <b>not</b> a reliable 'zeroed' indicator: the arm acknowledges every zero command as successful (ACK instruction 0x75, is_set_zero_successfully = 1).")
    s += code("""
.venv/bin/python tools/gripper_setup.py status   can_left can_right   # read-only (the homing bit it prints is unreliable)
.venv/bin/python tools/gripper_setup.py zero     can_left can_right   # disables the FOLDED arm (with your consent), zero, waits for the arm's ACK, then a follow-test
.venv/bin/python tools/gripper_setup.py test     can_left can_right   # opens 3 mm and back (gripper only)
.venv/bin/python tools/gripper_setup.py diagnose can_left can_right   # read-only bundle for a support request
""")
    s += callout("ok", "Solved in software: the gripper input layer",
                 "These grippers only adopt a new position target while the command varies slowly and continuously: a target that jumps "
                 "and then stays constant freezes them wherever they are (only 2 of 9 held targets were reached), and fast frame-to-frame "
                 "flicker does not help. A <b>slow sine on top of a ramped command</b> does: with a 0.5 mm, 6 Hz sine on the target (6 Hz was the fastest sine that stayed reliable, 25 Hz and up fails; no ramp needed), "
                 "9 of 9 held targets were reached within about 0.5 mm, over the full 0-70 mm stroke, with no measurable finger jiggle. "
                 "This layer (<font name='Mono'>piper_teleop/gripper_layer.py</font>) sits between every gripper target (Quest trigger, teach "
                 "pendant, a future policy) and the raw CAN command. It is on by default; the settings are <font name='Mono'>gripper_slew_mm_s</font>, "
                 "<font name='Mono'>gripper_dither_mm</font> and <font name='Mono'>gripper_dither_hz</font> in config/teleop.yaml. "
                 "Plots and videos of before/after: docs/media/. The root cause is still in the gripper firmware - worth reporting to AgileX "
                 "(docs/AgileX_support_request.md) - but it no longer blocks use.")
    s += P("The teleop server prints a warning at start-up if a gripper reports a fault, and the dashboard shows each gripper's commanded vs measured opening, with a warning if it sits "
           "still far from the command for 3 seconds. In teleop the grippers <b>start open</b>; squeezing the trigger closes them proportionally and releasing it opens them again. "
           "If tracking drops they keep their last command (they do not let go of what they hold).")
    s += P("For the pendant, <font name='Mono'>--scale</font> (100-200) stretches the pendant's travel to the follower's gripper "
           "(handy if the pendant feels cramped) and <font name='Mono'>--friction</font> (1-10) sets how stiff it feels.")

    # ------------------------------------------------------------------ 8
    s += [H1("8. Roles: leader or follower"),
          P("A Piper stores a role. The default is <b>follower</b> (0xFC, 'motion output arm'): it takes commands and reports feedback. "
            "A <b>leader</b> (0xFA, 'teaching input arm') does the opposite: it is free to drag, it broadcasts its own joint "
            "angles as command frames, and it sends <b>no joint feedback</b>."),
          ] + bullets([
        "Quest teleop: both arms must be <b>followers</b>.",
        "Arm-to-arm teleop: one leader, one follower (<font name='Mono'>tools/set_role.py</font>).",
        "If an arm 'has no feedback' after you used it as a leader, set it back: "
        "<font name='Mono'>.venv/bin/python tools/set_role.py can_left follower</font>, then power-cycle it."])
    s += callout("warn", "Unplug the leader before commanding a follower from a program",
                 "AgileX's dual-arm note: if a leader is still connected to the follower's bus, the leader keeps sending commands "
                 "and the follower can move 'out of control' even when your program is idle. In the PC-driven modes the arms are on separate buses, which avoids this.")

    # ------------------------------------------------------------------ 9-10
    s += [H1("9. Firmware, SDK version and joint limits"),
          P("The software clamps every command to the intersection of the SDK's limits and the manual's, which is tighter than either alone:")]
    s += table([["Joint", "Software limit (deg)", "Manual range (deg)", "Max speed (deg/s)"],
                ["J1", "-150 ... +150", "-154 ... +154", "180"], ["J2", "0 ... 180", "0 ... 195", "195"],
                ["J3", "-170 ... 0", "-175 ... 0", "180"], ["J4", "-100 ... +100", "-106 ... +106", "225"],
                ["J5", "-70 ... +70", "-75 ... +75", "225"], ["J6", "-100 ... +100", "+-100 (SDK says +-120)", "225"]],
               [20, 45, 60, 45])
    s += [H1("10. Shutting down properly")] + steps([
        "Stop the teleop program with Ctrl-C. By default it <b>parks</b> the arms slowly (0.5 rad/s) to the folded zero pose and leaves the motors <b>enabled and holding</b> "
        "(add <font name='Mono'>--release-on-exit</font> to power them down). Do not touch them while they move.",
        "Confirm the teach-button light is off.",
        "Switch the supplies off. If the arm ever has to be powered down in an unusual pose, support it by hand first."])

    # ------------------------------------------------------------------ 11
    s += [H1("11. Cameras")]
    s += P("The arms do not need cameras for teleoperation by sight. For recording or for wrist/overview views, plug cameras into the PC:")
    s += bullets([
        "<font name='Mono'>sudo apt install v4l-utils; v4l2-ctl --list-devices</font> lists them. Put high-resolution USB cameras on separate "
        "USB controllers (different ports on different hubs) - two 1080p streams on one hub starve each other.",
        "A wrist camera on a flange-side bracket moves with the gripper (best for close manipulation). A fixed overview camera "
        "behind and above the arms is the natural second.",
        "<b>Honest status:</b> this project does not yet stream camera video into the headset or record episodes. The WebXR page has a "
        "passthrough option so you can simply look at the real robot, which is the lowest-latency 'camera' available."])

    # ------------------------------------------------------------------ 12
    s.append(PageBreak())
    s += [H1("12. Troubleshooting")]
    s += table([["Symptom", "Likely cause", "What to do"],
                ["can_left does not exist", "Adapter not plugged / driver missing / not renamed", "scripts/can_setup.sh list; sudo modprobe gs_usb; scripts/can_setup.sh all"],
                ["SEND_MESSAGE_FAILED (100017) (SDK error)", "CAN interface down, bad wiring, or arm unpowered", "Check wiring and power; power-cycle the arm (AgileX's own advice)"],
                ["hw_check: no joint feedback", "No power; CAN_H/CAN_L swapped; adapter on wrong port; arm is set as leader", "Check 24 V and status light; swap wires; run tools/set_role.py ... follower"],
                ["Enable times out", "Arm in an error state / emergency stop latched / teach light on", "Teach light off; power-cycle; call MotionCtrl_1(0x02,0,0) (the software does this on enable)"],
                ["Gripper reads 0.0 mm", "A DISABLED gripper reports 0.0 regardless of where the fingers are; an enabled one reports the real opening. Closed + enabled also reads about 0"],
                ["Motors say 'enabled' but the arm ignores commands (nudge shows +0.00 deg)", "Arm not in CAN control mode (standby/teaching), or in an error/e-stop state, or still a leader", "Run hw_check.py ... --enable and read the 'arm says' lines: Control Mode must be CAN (0x1) and Arm Status NORMAL. Teach light off; power-cycle with the light off; set follower. The software now refuses to continue and prints this status if the arm will not enter CAN mode"],
                ["Gripper reads 0, won't move", "Stroke range not set", "tools/gripper_setup.py params --range 70, then zero"],
                ["Arm drops when software starts", "A reset/enable/mode frame reached an arm that was raised (earlier versions sent a reset and a fresh enable every start; suspected cause of a drop at the start of a second run)", "Current software adopts a healthy enabled arm without sending anything, and refuses state changes on a raised arm. If an arm still drops, note exactly WHEN (at connect, at the Enter prompt, during the home move, or after the program exited)"],
                ["Both arms mirror each other / wrong arm moves", "can_left and can_right swapped", "Re-do the USB-port mapping in config/can_ports.conf"]],
               [45, 55, 70])
    s += [H1("Sources"), P("AgileX PiPER Robotic Arm Quick Start User Manual V1.0 (2024.09): "
                           "static.generation-robots.com/media/agilex-piper-user-manual.pdf", "small"),
          P("AgileX piper_sdk (MIT licence): github.com/agilexrobotics/piper_sdk - README, assets/double_piper.MD, assets/Q&amp;A.MD, assets/can_config.MD", "small"),
          P("Meta Quest scriptable testing (adb proximity broadcasts): developers.meta.com/horizon/documentation/unity/ts-scriptable-testing/", "small")]
    doc.build(s)


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "Piper_Setup_Guide.pdf")
