from pdfkit import *  # noqa: F401,F403


def build(path):
    doc = Doc(path, "Piper teleoperation - user guide", "Piper teleoperation user guide")
    s = Story()
    s += [SP(30), P("Teleoperation user guide", "title"),
          P("Arm-to-arm, Quest 3 with the headset on, and Quest 3 controllers only (headless) - and how to make "
            "each as low-latency as the hardware allows.", "sub")]
    s += callout("warn", "Status: simulated and unit-tested, not yet run on real arms or a real Quest",
                 "Everything here was tested in a simulator, with 28 automated tests, against the SDK's own kinematics, and through a real "
                 "Chrome browser talking to the real server. It has <b>not</b> been run on physical Pipers or a physical Quest 3 - they were "
                 "still in their boxes. Chapter 10 lists exactly what is proven and what is not. Do the first real runs slowly, with the "
                 "power switch in reach, exactly as the checklist in chapter 7 says.")

    # ---------------------------------------------------------------- 1
    s += [H1("1. Choose a mode")]
    s += table([["Mode", "What you do", "PC in the loop?", "Best for"],
                ["A  Native linkage", "Move the leader arm (with the teach pendant); the follower copies it. Arms linked by AgileX's own CAN master/slave mode", "No", "Lowest latency, nothing to run. Try it first as your 'zero software' reference feel"],
                ["B  Software leader-follower", "Same, but each arm has its own CAN bus and the PC relays (<font name='Mono'>piper_teleop.leader_follower</font>)", "Yes", "Safety limits, aligned start, gripper scaling, later: recording"],
                ["C  Quest, headset on", "Two Touch controllers drive both arms; optional passthrough so you see the real robot", "Yes", "Bimanual teleop"],
                ["D  Quest, headless", "Headset sits on a stand with its screen kept awake; you use only the controllers", "Yes", "Teleop without wearing a headset (look at the robot directly)"]],
               [30, 78, 22, 40])
    s += [H3("Where latency comes from"), latency_diagram(), SP(4)]
    s += P("Chapter 6 goes through each box and the setting that changes it. A rule of thumb for feel: a pose-to-motion delay "
           "below about 50 ms feels direct; above 100 ms people start overshooting. Measured so far, on the PC side only: "
           "<b>~7 ms</b> from a pose packet arriving to the arm command being issued (simulation, default settings, which include a 10 ms smoothing filter). "
           "The Quest, the network and the real arm add to that and are what chapter 6 helps you measure.")

    # ---------------------------------------------------------------- 2
    s += [H1("2. Dry run on the PC (no hardware at all)"),
          P("Do this first. It runs the full pipeline - server, kinematics, safety logic, dashboard - against simulated arms.")]
    s += code("""
cd piper_meta_teleop
.venv/bin/python -m piper_teleop.quest_server --sim        # terminal 1
xdg-open http://localhost:8788/dashboard                    # PC dashboard: both arms drawn live
.venv/bin/python tools/fake_quest.py                        # terminal 2: pretend Quest, hands drawing circles
.venv/bin/python -m piper_teleop.leader_follower --sim      # separate test of mode B
.venv/bin/python -m pytest -q tests                         # 28 tests
""")
    s += P("In the dashboard you should see both arms go to the home pose, then the fake hands grab (state 'engaged') and the arms trace "
           "small circles. <b>SPACE freezes the arms</b> (in this terminal, or in the dashboard window). The page at <font name='Mono'>http://localhost:8788/?fake=1</font> does the same from "
           "a browser tab, exercising the real Quest page code.")

    # ---------------------------------------------------------------- 3
    s += [H1("3. Arm to arm")]
    s += [H2("Mode A - AgileX native linkage (no PC)")]
    s += P("From the SDK's dual-arm document: both arms connect to a CAN module with the leader's and follower's CAN lines joined; "
           "one arm is set to 0xFA (leader: 'teaching input arm'), the other to 0xFC (follower: 'motion output arm'). The leader then broadcasts "
           "its pose as ordinary control frames and the follower executes them in joint mode with high-follow. The roles can be set with the "
           "AgileX host-computer software (Windows) or from this project:")
    s += code("""
.venv/bin/python tools/set_role.py can_left  leader       # the arm you will move by hand (fit the teach pendant on its flange)
.venv/bin/python tools/set_role.py can_right follower
""")
    s += steps(["Fold both arms to the zero pose. Teach-button lights off.",
                "Join the two arms' CAN_H and CAN_L (and connect that bus to a CAN module if you want to watch it from the PC). "
                "AgileX's diagram in <font name='Mono'>assets/double_piper.MD</font> shows the wiring.",
                "Power the <b>follower first</b>, then the leader; wait a few seconds.",
                "Move the leader gently; the follower should follow. The pendant's squeeze controls the follower's gripper.",
                "If it does not follow: check the wiring and power-cycle both arms (AgileX's own advice)."])
    s += callout("warn", "Going back to PC-driven modes",
                 "Reset the leader first: <font name='Mono'>tools/set_role.py can_left follower</font> and power-cycle. Unplug the link between "
                 "the arms: a connected leader keeps sending commands to a follower even when your program is idle, so the follower can move "
                 "'out of control'. Put the arms at the zero pose before changing from link mode to CAN control (manual 2.2).")
    s += [H2("Mode B - software leader/follower")]
    s += P("Each arm has its own adapter (<b>can_leader</b>, <b>can_follower</b> in <font name='Mono'>config/can_ports.conf</font>, set up as in the "
           "setup guide). The leader is set to 0xFA once; the PC only <i>reads</i> it - the leader's motors are never enabled, so it stays free to drag.")
    s += code("""
.venv/bin/python tools/set_role.py can_leader leader
.venv/bin/python tools/set_role.py can_follower follower
.venv/bin/python -m piper_teleop.leader_follower            # --yes skips the confirmation; --source msgs if needed
""")
    s += bullets([
        "<b>Aligned start:</b> if the arms start in different poses the follower first moves to the leader at no more than 0.5 rad/s, "
        "and asks for Enter before doing so if the gap is over 10 degrees. Then it follows at up to 3 rad/s (<font name='Mono'>leader_follower.max_joint_speed</font>).",
        "<b>Silent leader:</b> if command frames stop for 150 ms (unplugged, powered off) the follower holds its last command.",
        "<b>Gripper:</b> the pendant/gripper opening is copied to the follower, multiplied by <font name='Mono'>gripper_scale</font>.",
        "<b>Ctrl-C</b> parks the follower slowly in the zero pose and leaves it enabled and holding (<font name='Mono'>--no-park</font> to skip the park, "
        "<font name='Mono'>--release</font> to power it down).",
        "If the follower never gets frames ('no command frames from the leader'), the leader may be in plain drag-teach instead of 0xFA: try "
        "<font name='Mono'>--source msgs</font>, which reads its joint <i>feedback</i> instead of its broadcast commands."])

    # ---------------------------------------------------------------- 4
    s.append(PageBreak())
    s += [H1("4. Quest 3 teleoperation, headset on (mode C)")]
    s += [H2("4.1 One-time Quest preparation")]
    s += steps([
        "Enable <b>Developer Mode</b> for the headset in the Meta Horizon phone app (Devices, your headset, Headset settings, Developer mode). "
        "Meta may ask you to set up a free developer account first; the exact menu names change between app versions.",
        "On the PC: <font name='Mono'>scripts/quest_adb.sh check</font> with the headset connected by a USB-C <b>data</b> cable. "
        "Put the headset on and accept <i>Allow USB debugging</i> (tick 'always allow').",
        "That is all - there is nothing to sideload. The controller extractor is a web page the PC serves (WebXR in Quest Browser); "
        "it reads both controllers' poses, trigger, grip, sticks and buttons and streams them as 62-byte binary packets. "
        "A native sideloaded app would not read controllers any faster than the browser's frame loop, and it would cost you a build toolchain."])
    s += [H2("4.2 Link A: USB (recommended - most stable, lowest jitter)")]
    s += code("""
.venv/bin/python -m piper_teleop.quest_server --usb           # real arms; starts the server, THEN sets up the USB link
                                                              # and opens the page in the Quest (adb reverse + browser)
""")
    s += callout("warn", "Do not chain the server and the adb script with a semicolon",
                 "The server runs in the foreground, so <font name='Mono'>server; scripts/quest_adb.sh usb</font> only runs the adb script <b>after you quit the "
                 "server</b>. Use <font name='Mono'>--usb</font> (above), or run the adb script from a second terminal while the server is up. "
                 "The port is the one in <font name='Mono'>config/teleop.yaml</font> (<font name='Mono'>transport.port</font>, default 8788); the adb script reads it from there, "
                 "so do not pass a different one.")
    s += P("<font name='Mono'>adb reverse</font> makes the Quest's <font name='Mono'>localhost:8788</font> reach your PC through the USB cable. "
           "A page on <font name='Mono'>http://localhost</font> counts as a secure context, which WebXR requires - no certificate needed. "
           "If the browser did not open by itself, type <font name='Mono'>localhost:8788</font> into the address bar.")
    s += [H2("4.3 Link B: Wi-Fi (no cable)")]
    s += code("""
scripts/make_cert.sh                                          # self-signed cert incl. your LAN IPs (once)
.venv/bin/python -m piper_teleop.quest_server --tls
# Quest Browser:  https://<PC-IP>:8788/   ->  Advanced -> Proceed (once)
""")
    s += bullets([
        "PC on <b>Ethernet</b> to the router; Quest on that router's <b>5 or 6 GHz</b> band, in the same room, no mesh hop, no VPN (a Tailscale address is fine for reaching it but adds latency).",
        "The page shows the live round-trip time. Wi-Fi is usually a few ms but with spikes; if you see spikes above ~20 ms, use the cable.",
        "Plain <font name='Mono'>http://&lt;PC-IP&gt;</font> will not work: WebXR needs HTTPS off-localhost."])
    s += [H2("4.4 In the headset")]
    s += callout("note", "The dashboard tells you what the headset is doing",
                 "The <b>Quest headset</b> card at the top of <font name='Mono'>http://localhost:8788/dashboard</font> shows what the headset page reports (page connected? "
                 "VR session? which controllers are tracked? errors from inside VR) and says in plain words what to do next. "
                 "'<i>no controller data</i>' on an arm just means no pose packet has arrived: either the page is not open, or <b>nobody pressed Enter VR</b> "
                 "(pose data only flows inside a running VR session), or <b>the headset is asleep</b> (a Quest that is not on your head sleeps within seconds and drops the connection - "
                 "wear it, or use headless mode, chapter 5), or it sees your hands instead of the controllers.")
    s += steps([
        "Tick <b>Passthrough</b> (so you see the real arms) - or untick for a plain VR background that turns green while you are engaged and red when the arms are frozen.",
        "Press <b>Enter VR &amp; start streaming</b> with the controller ray. The first time, accept Quest Browser's immersive-session prompt.",
        "The arms (if real) move to the home pose after you confirmed in the terminal. Stand facing the robot.",
        "<b>Axis calibration</b> starts by itself once both controllers are tracked (the arms stay frozen until it is done or skipped). You are told what to do "
        "by voice in the headset (best effort), on the dashboard, and by haptics: <b>squeeze both GRIPs, push the controllers about 20 cm FORWARD, let go</b>; "
        "then the same <b>to your RIGHT</b>, then <b>UP</b>. Each move is checked for length and straightness; a strong buzz means 'try again'. "
        "It also catches left/right mix-ups and tilted moves. Stand where you will work, facing the table. Both thumbsticks skip or cancel it.",
        "Then hold <b>GRIP</b> on a controller to make that arm follow that hand. Release to freeze it (it stops where it physically is). "
        "Squeeze <b>TRIGGER</b> to close the gripper."])
    s += table([["Control", "Action", "Notes"],
                ["GRIP (side button), hold", "Arm follows that hand", "Relative: the arm never jumps when you grab. Needs a fresh press after any tracking loss or start-up"],
                ["TRIGGER", "Gripper: released = open (70 mm), squeezed = closed (proportional)", "Grippers START OPEN. Work whenever the hand is tracked, with or without GRIP. Each gripper must have been zeroed once (setup guide ch. 7)"],
                ["A / X", "Fine mode on/off (motion x 0.4)", "Rescales on the fly without a jump"],
                ["B / Y", "Send that arm to the home pose", "Only when GRIP is not held; grabbing cancels it"],
                ["Click both thumbsticks", "Start the guided axis calibration (or cancel / skip it while it runs)", "Also: dashboard 'Calibrate axes', console key c; x or the Skip button cancels"],
                ["SPACE (server terminal, no Enter needed; or dashboard window)", "FREEZE: every arm holds where it physically is. Software hold only: no power cut, gripper untouched, no firmware stop", "Also the red FREEZE button on the Quest page. Resume (r / dashboard button) re-arms; you must then release and press GRIP again"]],
               [38, 60, 72])
    s += P("<b>Haptics</b> (feel instead of look): short firm tick = engaged; tiny tick = fine mode toggled; soft buzz = the target is at the arm's reach limit; "
           "long strong buzz on both = FREEZE, a control-loop error or a failed calibration; 80 ms strong buzz = heading calibrated.")
    s += callout("note", "Status squares in your view",
                 "Two small squares float at the bottom of your view (left arm | right arm), drawn even in passthrough, so a dark screen can never hide what is happening: "
                 "<b>grey</b> = that controller is not tracked, <b>amber</b> = tracked, waiting for GRIP, <b>green</b> = the arm is following you, "
                 "<b>blue</b> = homing/parking, <b>red</b> = frozen. A <b>white dot</b> inside a square means the browser itself sees that GRIP pressed - "
                 "so if you squeeze and get no white dot, the problem is on the headset side; white dot but amber square means the PC is not engaging "
                 "(read the arm's status on the dashboard).")
    s += P("Keep the PC dashboard open on a second screen (<font name='Mono'>http://localhost:8788/dashboard</font>): loop rate, stream rate and gaps, "
           "RTT, each arm's state, and tracking error.")

    # ---------------------------------------------------------------- 5
    s += [H1("5. Quest 3 controllers only, no headset on your head (mode D)")]
    s += P("A VR session only lives while the headset believes it is on a head; when you take it off it goes <b>visible, blurred, hidden, session ended</b> within about two "
           "seconds (this is exactly what the session log showed on the first attempt). Headless mode makes the headset believe it is still worn. The teleop server now does "
           "this <b>for you, automatically</b>, at the right moment - you do not type anything while wearing the headset.")
    s += [H2("5.1 The recipe")]
    s += steps([
        "Start the server with <font name='Mono'>--usb</font> (as in chapter 4) and open the page in the headset.",
        "Tick <b>Headless mode</b>, then press <b>Enter VR &amp; start streaming</b> while <b>wearing</b> the headset. A human click is required to start any VR session.",
        "The moment you press the button the server sends Meta's documented proximity override (<font name='Mono'>prox_close</font>), <b>checks through the headset's own "
        "property</b> that it took effect, and confirms with a <b>triple strong buzz on both controllers</b> (plus a voice message if speech works in your browser): <b>wait for the buzz</b> before taking the headset off - on one attempt the headset was taken off before the override had landed.  The dashboard's Quest card shows "
        "'headless override: ON'.",
        "Take the headset off and <b>stand it upright</b> on a stand or tripod, lenses facing the room, about chest height, 0.5-1 m in front of where your hands will be, room lit. "
        "<b>Its cameras are what track the controllers</b>, so both controllers must stay inside what it can see (hands in front of your chest).",
        "Let the axis calibration run (chapter 4), then use GRIP / TRIGGER as usual. Look at the robot, not the headset.",
        "Ctrl-C the server when finished: it <b>restores the headset's normal behaviour</b> by itself and prints 'headset proximity behaviour restored'. "
        "(Manual control: <font name='Mono'>scripts/quest_adb.sh headless-on | headless-off | headless-status</font>; each verifies the result.)"])
    s += callout("warn", "'Finding position in room' - the dialog that kills headless sessions",
                 "With the override on, a headset that <b>cannot localise itself</b> (face-down, lenses blocked, dark room) puts the system dialog "
                 "<i>'Finding position in room - your headset can't detect your movement'</i> in front of every app, which blurs and ends the VR session. This was reproduced on a real Quest 3. "
                 "It is a placement problem: stand the headset <b>upright</b> with the lenses facing a lit room so its cameras can see features and your controllers. The dialog then clears by itself. "
                 "The dashboard detects and names it (the 'headless override' row and the hint line); the <font name='Mono'>headless-status</font> command prints the dialog text. "
                 "Do <b>not</b> press 'Continue without tracking': the controllers are tracked by the headset's cameras, so they would not work.")
    s += callout("warn", "Caveats",
                 "<b>Heat, battery, burn-in:</b> a headset that never turns its display off runs warm and drains; the page keeps the display black in headless mode. Do not leave it like this unattended, "
                 "and make sure the server restored it afterwards (<font name='Mono'>headless-status</font> must show <font name='Mono'>set_proximity_close=false</font>). "
                 "<b>Tracking region:</b> controllers outside the headset's camera view are IMU-estimated only; the page flags that as 'emulated', the server treats it as tracking lost and the arm "
                 "<b>holds</b> until you release and grab again. <b>Auto-sleep:</b> if the session still drops after a long idle, Meta's <font name='Mono'>disable_autosleep</font> property needs your Store PIN "
                 "and is deliberately not scripted. <b>If the headset is rebooted or the cable pulled</b> the override is lost: wear it and press Enter VR again.")
    s += callout("note", "Fallback if the adb override misbehaves",
                 "A widely reported alternative is covering the headset's proximity sensor (between the lenses) with a small piece of opaque tape so it always reads 'worn'. "
                 "That is a physical hack you can undo. A person must still press Enter VR once.")

    # ---------------------------------------------------------------- 6
    s += [H1("6. Getting the lowest latency")]
    s += table([["Stage", "What adds delay", "What to do"],
                ["Quest tracking + frame", "Poses arrive once per display frame (72/90/120 Hz). 120 Hz = 8.3 ms between samples, 72 Hz = 13.9 ms", "The page asks for the highest supported rate automatically; check the 'XR started, frame rate N Hz' line on the page"],
                ["Network", "USB: small and steady. Wi-Fi: small on average, spiky", "Use USB (4.2). On Wi-Fi: Ethernet PC, 5/6 GHz, one room. Page shows RTT; the server drops, never queues, stale packets"],
                ["PC loop", "Control loop is 200 Hz (up to 5 ms wait for the next tick), plus the smoothing filter", "Defaults are now <font name='Mono'>smoothing_ms: 3</font>, <font name='Mono'>filter_min_cutoff: 5</font>, <font name='Mono'>filter_beta: 60</font> (measured PC-side lag in simulation 15 ms, down from 35 ms with the old 10 / 3 / 12). Lower smoothing or raise cutoff/beta for even less lag (more jitter)"],
                ["Speed limiting", "<font name='Mono'>max_joint_speed</font> clamps fast moves (adds lag when you move quickly)", "Default 2.0. Raise toward 2.5-3 rad/s once you trust it (arm max is ~3.1-3.9 rad/s)"],
                ["Arm firmware", "Standard joint mode plans and smooths its own trajectory", "<font name='Mono'>piper.follow_mode: high_follow</font> (CAN code 0xAD) removes the arm's own smoothing. <b>Lowest latency but it chases targets hard</b> - try only at low speed limits, hand near the switch"],
                ["PC scheduling", "Browser/other apps stealing CPU; frequency scaling", "Close heavy apps; <font name='Mono'>sudo cpupower frequency-set -g performance</font>; dashboard 'worst step' should stay at a ms or two"],
                ["Gripper", "Gripper has its own response time", "Nothing to tune in software; effort is a trade-off with safety"]],
               [28, 66, 76])
    s += H3("How to measure the whole chain")
    s += P("Film a controller and the arm together with a phone in 240 fps slow motion. Flick the controller (GRIP held), count frames between the controller starting to move "
           "and the arm starting to move, and divide by 240. The dashboard's RTT shows only the network round trip; the 'pose stream' row shows the packet rate and the worst gap between packets "
           "(a long gap is a hiccup you can feel).")

    # ---------------------------------------------------------------- 7
    s += [H1("7. Safety: checklist and what the software does")]
    s += callout("danger", "Before the first real run",
                 "1. Clear workspace, nobody within reach, loose clothes and hair secured. 2. Hand on the power switch. "
                 "3. Both teach-button lights OFF. 4. Both arms folded at the zero pose before powering up. "
                 "5. <font name='Mono'>tools/hw_check.py ... --enable --nudge</font> passed on each arm. 6. First run with "
                 "<font name='Mono'>max_joint_speed: 0.8</font> and <font name='Mono'>follow_mode: standard</font>. 7. Check "
                 "<font name='Mono'>workspace.z[0]</font> is above your table surface and <font name='Mono'>tcp_offset</font> matches your gripper (flange to fingertip, in metres).")
    s += table([["Situation", "What happens"],
                ["Start-up on an arm that is already raised, enabled and holding", "It is <b>adopted untouched</b>: no reset, enable or mode frame is sent (the startup log says 'ADOPTED'). Anything that would change the state of a raised arm (leaving teaching mode, clearing an e-stop) is <b>refused</b> with an explanation"],
                ["Release GRIP (the deadman)", "That arm stops where it physically is (not where it was last told to go) and holds. TRIGGER only ever works the gripper and never moves the arm"],
                ["GRIP held while starting, or when tracking returns", "Ignored until you release and press again (no surprise motion)"],
                ["Controller leaves the headset's view / 'emulated' pose", "Treated as lost: arm holds"],
                ["No packets for 150 ms (Wi-Fi drop, page closed)", "Arm holds (<font name='Mono'>stale_ms</font>)"],
                ["Target outside reach / singular", "The arm slides along the edge of what it can reach instead of freezing, a soft buzz fires, and the arm status says which way: 'too far out', 'too close to its own base', 'too far sideways', 'too high', 'too low'. The offset between hand and arm is re-based (anti-windup), so overshooting by 30 cm no longer leaves a 21 cm dead zone to travel back through - now about 2 cm. Only targets more than 15 cm out of reach are held"],
                ["Target outside workspace box", "Clamped to the box (never pulled away from where the TCP already is)"],
                ["Fast hand motion", "Joint speed limit; the arm catches up smoothly"],
                ["Software error in the control loop", "All arms FREEZE where they are, loop exits, error shown on the dashboard"],
                ["SPACE / FREEZE", "All arms stop at their current physical pose and keep actively holding it (joint commands continue; gripper untouched; motors stay enabled). Stays frozen until Resume. Optional <font name='Mono'>control.estop_mode: firmware</font> also sends the arm's fast-stop frame - unverified on your arms, so not the default"],
                ["Ctrl-C / quit", "Arms are parked slowly at the folded zero pose and stay enabled and holding. Motors are only disabled with --release-on-exit"],
                ["<b>PC crash or power loss</b>", "<b>The arm is NOT stopped by software.</b> It keeps its last command. SPACE is a software freeze; the only real emergency stop is the power switch"]],
               [70, 100])

    # ---------------------------------------------------------------- 8
    s += [H1("8. Configuration reference (config/teleop.yaml)")]
    s += table([["Key", "Default", "Meaning"],
                ["defaults.pos_scale / rot_scale", "1.0 / 1.0", "Hand motion to TCP motion. 0.5 = half-speed, finer work"],
                ["defaults.fine_factor", "0.4", "Extra scale while fine mode (A/X) is on"],
                ["defaults.max_joint_speed", "2.0 rad/s", "Speed cap while engaged"],
                ["defaults.anti_windup", "0.85", "How much of an overshoot past the arm's reach is absorbed so no dead zone builds up (0 = off)"],
                ["defaults.ik_max_pos_err", "0.15 m", "Only targets farther than this out of reach make the arm hold still (was 5 cm and froze it too often)"],
                ["defaults.smoothing_ms / filter_min_cutoff / filter_beta", "3 / 5 Hz / 60", "Latency vs smoothness of the hand signal (see chapter 6)"],
                ["defaults.gripper_slew_mm_s / gripper_dither_mm / gripper_dither_hz", "0 / 0.5 / 6.0", "Gripper input layer: the trigger (or a policy) sets a target; the raw gripper command is the target plus a 0.5 mm, 6 Hz sine (6 Hz = fastest sine that stayed reliable; 25 Hz and up fails). No ramp was fastest (~0.25 s) and most reliable; set gripper_slew_mm_s (e.g. 150) only for gentler closing. Without the sine these grippers freeze on held targets. 0.5 mm showed no measurable finger jiggle"],
                ["defaults.rot_axis_scale", "[0.5, 1.0, 0.3]", "Hand rotation to gripper rotation gains: roll, pitch, yaw. Yaw/roll are expensive for a Piper wrist; 1 = copy the hand exactly"],
                ["defaults.wrist_pitch_min_deg", "20", "J5 floor: keeps the wrist away from the singularity where J4/J6 spin against each other"],
                ["defaults.wrist_posture_weight", "[0.2, 0.05, 0.2]", "Pull of J4/J5/J6 toward the home values. 0 = off; higher = calmer wrist, less exact gripper orientation"],
                ["defaults.stale_ms", "150", "Hold when data is older"],
                ["defaults.tcp_offset", "[0,0,0.12]", "Flange to fingertip (m). Wrong value = rotation pivots about the wrong point. Measure it"],
                ["defaults.workspace", "x 0.05-0.32, y +-0.30, z 0.05-0.40", "TCP box in each arm's base frame (m). Facing arms: x max below separation/2 - 0.05"],
                ["defaults.home_deg", "[0,35,-34,-4,37,10]", "Start pose: folded rest pose with the TCP 5 cm forward and 10 cm up, tool tilted 15 deg further down than at rest (so sideways motion works)"],
                ["arms.left/right.base_yaw_deg", "-90 / +90 (facing arms)", "Arm forward axis vs operator forward, ccw. Side by side facing the same way: 0 / 0. Swap the signs if an arm moves the wrong way"],
                ["piper.driver / piper.firmware", "pyagxarm / auto", "pyagxarm = AgileX's current library pyAgxArm (default; auto picks the firmware profile, v189 on S-V1.8-9). piper_sdk = the older library this project used first"],
                ["config/joint_lock.yaml", "can_left until 2026-10-28 13:00", "Time-limited JOINT LOCK per arm: only gripper and read-only frames reach it; every joint, mode, enable/disable, reset and e-stop frame is refused at the CAN-send layer of both libraries. Teleop runs gripper-only on a locked arm. Does not cover programs outside this project"],
                ["piper.follow_mode", "standard", "high_follow = lowest latency, riskiest (pyagxarm: move_js)"],
                ["piper.gripper_effort", "1000", "0.001 N.m units, 0-5000"],
                ["piper.dh_is_offset", "1", "0 only for firmware older than S-V1.6-3"],
                ["control.rate_hz", "200", "Control loop; the SDK demos use 200 Hz"],
                ["defaults.orientation_mode", "follow_base", "follow_base: gripper heading follows the arm's swing around its base (sideways motion needs only J1). world: gripper keeps the hand's world orientation"],
                ["calibration.auto", "true", "Run the guided axis calibration by itself and keep the arms frozen until it is done or skipped"],
                ["control.estop_mode", "freeze", "freeze = SPACE holds the pose in software. firmware = also send the arm's fast-stop"],
                ["arms.left/right.can / hand", "can_left / left", "Which adapter and which controller"],
                ["leader_follower.*", "see file", "CAN names, speed cap, gripper scale, leader source"]],
               [48, 40, 82])

    # ---------------------------------------------------------------- 9
    s += [H1("9. Troubleshooting")]
    s += table([["Symptom", "Check"],
                ["Quest page will not load over USB", "scripts/quest_adb.sh check; re-run scripts/quest_adb.sh usb (adb reverse is lost when the cable is unplugged); is the server running on 8788?"],
                ["'Enter VR' disabled / 'WebXR not available'", "Address must be http://localhost:PORT (USB) or https://... (Wi-Fi). Plain http:// to a LAN IP is blocked"],
                ["adb shows 'no permissions'", "scripts/quest_udev.sh (once; needs sudo), unplug/replug the cable, scripts/quest_adb.sh check. Quest 3 is USB id 2833:5013"],
                ["Arm barely moves sideways, 'IK rejected' (custom home pose)", "A tool held nearly horizontal at the start is near a wrist singularity. Tilt the home pose further down, as the default does"],
                ["Arms hit each other", "Lower workspace.x[1] for facing arms, check base_yaw_deg, increase base separation (setup guide, chapter 3)"],
                ["Server says 'port ... already in use'", "Another program has it (ss -ltnp | grep :PORT). Use --port N or transport.port in config/teleop.yaml; scripts/quest_adb.sh usb reads the port from the config (or: usb --port N). Nothing is started and no arm is touched in that case"],
                ["Page says 'PC link: disconnected'", "Server not running, wrong port, firewall (allow 8788), or HTTPS cert not accepted yet"],
                ["Arms twist a lot (wrist J4/J6 swinging)", "Fixed by default since this version: J5 floor, wrist posture bias and gentler yaw/roll gains. In simulation a hand that wanders and rotates naturally used to swing J4 100 deg and J6 120 deg; now about 9 and 7 deg. If it is still too twisty: raise wrist_posture_weight (e.g. 0.4) or lower rot_axis_scale yaw/roll. If the gripper should follow your wrist more: do the opposite"],
                ["Gripper does not open/close / stops short", "The Piper gripper only adopts targets while the command varies slowly. The gripper input layer (piper_teleop/gripper_layer.py: 0.5 mm @ 6 Hz sine on the target, no ramp, defaults.gripper_slew_mm_s / gripper_dither_mm / gripper_dither_hz) fixes this: held targets reached 9/9 vs 2/9 without it, within ~0.5 mm. If it still misbehaves: tools/gripper_setup.py test can_left; check the dashboard's commanded vs measured opening"],
                ["Headless: the session ends as soon as I take the headset off", "Tick Headless BEFORE pressing Enter VR (the server arms the override only for headless sessions). Check the dashboard's 'headless override' row. If it says NOT ACTIVE or names the 'Finding position in room' dialog: place the headset upright facing a lit room. Manual check: scripts/quest_adb.sh headless-status. The session log (logs/session.log) shows visible -> visible-blurred -> hidden -> ended when the override was missing"],
                ["Screen goes black after Enter VR and nothing moves", "Look at the Quest headset card: 'frames/s' must be about 72-120. 0 means the browser is not delivering frames (a system dialog covering the app, or a stuck session): dismiss it, exit VR, press Enter VR again. The page reports this on its own after 5 s. Everything the page and server saw is also written to logs/session.log"],
                ["One arm often stops following / 'at reach limit'", "The status says which way (e.g. 'too close to its own base'). Each arm only has about 10 cm of travel toward its own base from the folded start pose, and outward is capped by workspace.x[1] (0.32 m for your 85 cm spacing). Move the hand the other way - the arm answers within ~2 cm. logs/teleop_*.csv records hand, target, reached position and error at 40 Hz for every engaged arm, so a specific complaint can be analysed from data"],
                ["Headless: arms hold for ~15 s after taking the headset off", "Normal: while you carry the headset to its stand the controllers are only guessed from motion sensors ('emulated'), so the arms hold. When both controllers are seen again you feel one short tick on each. Stand the headset upright with the lenses facing the room and keep the controllers in front of it"],
                ["Arms say 'no controller data'", "Read the Quest headset card on the dashboard. Page not connected: wake the headset, scripts/quest_adb.sh usb (or start the server with --usb). Page open but 'idle': press Enter VR & start streaming. Session running but nothing tracked: pick up the controllers (hand tracking has no buttons), keep them in view"],
                ["Controllers 'none tracked'", "Wake them (press a button), check batteries; in headless mode check the stand position"],
                ["Arm does nothing when I hold GRIP", "State must be 'idle' with a fresh press: release fully, then grip. Check the arm's own state in the dashboard (frozen? stale?)"],
                ["Arm moves the wrong way / sideways", "Run the axis calibration again (click both thumbsticks / dashboard 'Calibrate axes'). Facing arms: check base_yaw_deg (-90 left arm, +90 right arm); a mirrored result means swap the signs"],
                ["Moving the hand one way mostly swings the arm's base and barely moves the gripper", "Use orientation_mode: follow_base (default). 'world' locks the gripper's heading, so every sideways move needs J1 plus a large J4/J6 counter-rotation. In/out (toward the base) travel is limited to about 10 cm from the folded start pose"],
                ["Arm lags / feels rubbery", "Chapter 6 table; check dashboard 'worst step' and RTT"],
                ["Buzz and 'at reach limit' constantly", "Re-grab nearer the middle of your range; consider pos_scale below 1; raise workspace limits only if the arm can really go there"],
                ["Arm twitches at the edge of reach", "Normal near singularities; the IK is damped there. Move back or use fine mode"],
                ["'could not enable arm'", "Setup guide chapter 12; is it still set as leader?"],
                ["Everything frozen ('FROZEN' shown)", "Look at 'error' on the dashboard; fix; press Resume (then release/press GRIP)"]],
               [58, 112])

    # ---------------------------------------------------------------- 10
    s += [H1("10. What is verified and what is not")]
    s += table([["Verified here", "How"],
                ["Forward kinematics identical to AgileX's SDK (random poses, error ~1e-16 m); Jacobian vs finite differences; IK convergence", "Automated tests"],
                ["Frame mapping (hand forward/left/up to robot x/y/z), no jump on grab, release-hold, stale-data hold, re-grab rule, speed limit, workspace clamp, homing, freeze", "Automated tests with a simulated arm"],
                ["Binary protocol both ways (JavaScript encoder in a real Chrome to the Python decoder); HTTP, WS and HTTPS/WSS serving; malformed packets ignored", "Chrome headless + tests"],
                ["PC-side pose-to-command reaction ~7 ms; control loop 200 Hz", "End-to-end test over a real WebSocket"],
                ["SDK unit conversions (rad to 0.001 deg, metres to 0.001 mm, joint/gripper clamps, gripper enabled at current opening)", "Tests against a fake SDK interface"],
                ["Arm-to-arm: aligned start, speed cap, silent-leader hold", "Tests + simulator run"]],
               [120, 50])
    s += table([["NOT verified (needs your hardware)", "Why it matters"],
                ["Real CAN traffic, enable sequence, firmware V3/V2 behaviour, the 0xAD high-follow mode", "Sequences follow the SDK demos and docs but were never executed against an arm"],
                ["Native linkage (Mode A) wiring and role behaviour", "Taken from AgileX's documents"],
                ["Quest Browser WebXR session, the controller button mapping (trigger 0, grip 1, stick 3, A/X 4, B/Y 5), haptics, 120 Hz request, passthrough", "Standard WebXR 'xr-standard' mapping, not run on a headset"],
                ["adb commands: <font name='Mono'>adb reverse</font>, browser launch intent, <font name='Mono'>prox_close</font>/<font name='Mono'>automation_disable</font>", "Documented by Meta / community; the launch intent is a best guess - type the URL if it does not open"],
                ["Home pose and the 0.12 m TCP offset on your actual gripper", "Computed from the SDK's kinematics; check clearance on first run"],
                ["Whether the arm folds cleanly from zero pose to home along a straight joint-space path", "Watch the first homing move closely"]],
               [90, 80])
    s += [H3("Project layout"),
          ] + code("""
piper_teleop/   kinematics.py arm.py controller.py system.py
                quest_server.py leader_follower.py protocol.py
web/            index.html quest.js (Quest WebXR page)   dashboard.html (PC)
config/         teleop.yaml   can_ports.conf(.example)
scripts/        install.sh  can_setup.sh  quest_adb.sh  make_cert.sh
tools/          hw_check.py go_rest.py set_role.py
                gripper_setup.py fake_quest.py
tests/          28 automated tests
docs/           build_pdfs.py + the two PDFs
""")
    doc.build(s)


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "Teleop_User_Guide.pdf")
