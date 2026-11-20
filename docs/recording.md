# Recording a dataset (LeRobot v3.0 / v2.1, 30 Hz)

Record demonstrations while you teleoperate arm-to-arm, straight into a LeRobot dataset directory you can push to a repo and
train on (OpenPI, ACT, Diffusion Policy, ... anything that reads LeRobot). No torch and no `lerobot` install is needed to record.

    pip install -r requirements.txt        # pyarrow, av, opencv-python-headless are the recording dependencies

## Run it

    # set up the arms as for arm-to-arm teleop (docs/teleop_guide.py, mode B): leader in drag-teach, follower on CAN
    .venv/bin/python -m piper_teleop.record --root datasets/pick_cube --task "pick up the cube"
    .venv/bin/python -m piper_teleop.record --root datasets/pick_cube --format v2.1      # for OpenPI
    .venv/bin/python -m piper_teleop.record --root /tmp/try --sim --fake-camera          # no hardware: a rehearsal

Arm options are the same as `python -m piper_teleop.leader_follower` (`--mode`, `--j1-offset`, `--leader-arm`, `--follower-arm`,
`--source`, `--yes`...). Arms are always named by the **physical side**: by default you move the **right** arm (leader) and the
**left** arm (follower) copies it (`leader_follower.leader_arm / follower_arm` in the config).
Pointing `--root` at an existing dataset of the same format **appends** to it (same fps / features required, otherwise it
refuses and says why). A non-dataset, non-empty directory is refused.

Useful options: `--num-episodes N` (stop after N kept episodes), `--episode-time-s S` (end each episode by itself),
`--reset-time-s S` (start the next one by itself), `--no-review` (save each episode as soon as it ends), `--no-cameras`,
`--fps`, `--task`. Defaults live in the `record:` section of `config/teleop.yaml`.

## Keys

| state | keys |
|---|---|
| **WAIT** (arms follow, nothing is written) | `Enter` / `Space` / `s` start an episode, `q` quit |
| **REC** | `n` / `Enter` / `Space` finish, `p` pause, `r` discard and start over, `q` finish then quit |
| **PAUSED** (episode stays open, frames not written) | `p` / `Space` resume, `n` finish, `r` discard, `q` finish then quit |
| **REVIEW** (the finished episode is *not saved yet*) | `Enter` / `k` keep it, `r` retake, `d` discard, `q` keep it and quit |

An episode only reaches the dataset when you keep it. A retake throws the take away and goes back to WAIT so you can reset
the scene. Ctrl-C stops the session; an unfinished episode is discarded, kept ones are already on disk.
The recorder will not start an episode while the follower is still moving to the leader's pose.

## Homing before each episode

By default the robots go home before **every** episode (the first one, after a kept episode, after a retake):

1. the follower drives to the folded **rest** pose (all joints 0) at `record.home.speed` (0.5 rad/s) and waits there - it ignores the leader meanwhile;
2. the status line says what to do: `follower is home - move the leader to its home pose (23 deg off, need <= 10)`;  ("home" = the rest pose here)
3. once **you** have put the leader within `tol_deg` of that pose, the follower follows again and `Enter` starts the episode.

The leader is a hand-driven arm in drag-teach, so software cannot move it (and will not switch its mode or release its motors);
that is why step 2 is a gate on you rather than a motion. Every episode then starts from the same pose, which is what policies
like. `Enter` before step 3 is refused with the same message. Nothing is written while homing and the gripper is left alone.

    home: {enabled: true, deg: null, tol_deg: 10, speed: 0.5}       # record.home in config/teleop.yaml

`deg` is in the FOLLOWER's joint space (after the joint mapping); `null` = the folded rest pose `defaults.park_deg` (`[0, 0, 0, 0, 0, 0]`),
which the leader simply rests in, so it is easy to match. The raised IK start pose (`defaults.home_deg`) was the first choice and was
hard to match by hand; put its angles in `deg` if you ever want it.
Skip it with `--no-home` or `enabled: false`. The move is a straight line in joint space from wherever the last episode ended
(folding toward the base), so clear the area of anything in the way before you keep an episode - this is the same kind of move as the follower's
start-up alignment. In `--sim` the simulated leader puts itself at home.

## What is recorded (per frame, 30 Hz)

| feature | shape | meaning |
|---|---|---|
| `observation.state` | (7,) float32 | follower joints 1..6 [rad], gripper opening 0..1 (1 = fully open, 70 mm) |
| `action` | (7,) float32 | the leader's joints mapped onto the follower [rad] (after `mode` / `j1_offset_deg`, clipped to the joint limits), gripper target 0..1 |

Both are named after the physical arm the policy drives (the follower): `left_joint_1` ... `left_joint_6`, `left_gripper`.
| `observation.images.<name>` | (H, W, 3) video | one per camera in `record.cameras` (none right now) |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | | LeRobot bookkeeping |

The gripper is normalised because the pendant (16-105 mm) and the follower (0-70 mm) have different strokes; the mapping is
`leader_gripper_mm` in the config. Frames where the leader is silent are skipped (shown as `dropped`), as are slots the loop
missed - it never bursts to catch up. `STALE CAMERA` in the status line means a camera frame was older than 0.15 s.

## Left / right and CAN names

The CAN interface names come from the USB port an adapter sits in, not from the arm it is wired to, and the two adapters came
up swapped: the physical **left** arm is on `can_right` and the **right** arm on `can_left`. That mapping lives in ONE place,
`hardware.can` in `config/teleop.yaml`; everything else (arms, leader/follower, recording, dataset names) says left / right
meaning the physical arm. If you fix the udev rule (`config/can_ports.conf`, needs sudo) or the adapters swap again, change only
those two lines. Tools that take a CAN name (`hw_check.py`, `set_role.py` ...) still want the CAN name: the left arm is `can_right`.

## Cameras

`record.cameras` in `config/teleop.yaml`; the key is the feature name (`observation.images.<key>`) and follows the arm / the view:

    cameras:
      cam_high:       {type: orbbec, serial: CC1365303LL, width: 640, height: 480, fps: 30}   # top / front view
      cam_left_wrist: {type: orbbec, serial: CC1WC5200WW, width: 640, height: 480, fps: 30}   # mounted on the LEFT arm (follower)
      # cam_right_wrist: {type: opencv, index: 0, ...}                                       # any UVC webcam
      # type: fake                                                                            # test pattern

Pick the names before the first episode: a dataset can't change its camera set afterwards (appending checks this). An arm-less
rehearsal can use `--no-cameras` or `--fake-camera`.

The Orbbec DaBai needs the v1 SDK in the Python 3.10 camera venv, so it runs in a helper process (`tools/orbbec_stream.py`,
frames shared through `/dev/shm`). Each unit must enumerate at **480 Mb/s** (`cat /sys/bus/usb/devices/*/speed` for the 2bc5
devices): on a 12 Mb/s link it delivers no frames. Tested with both units at 640x480@30 together with the recorder: 30.0 Hz,
no dropped frames.

### Using the ASUS Xtion as the top camera

    .venv/bin/python -m piper_teleop.record --root datasets/pick_cube --cameras xtion_top

`--cameras NAME` swaps in `record.camera_sets.NAME` from the config. `xtion_top` makes `cam_high` the ASUS Xtion (PrimeSense,
USB `1d27:0601`, colour only, 640x480@30, opened through OpenNI2 by `tools/xtion_stream.py`) and leaves `cam_left_wrist` exactly
as it is. The feature names are the same as the default set, so the dataset layout does not change; only what `cam_high` looks
through does, and the Xtion's colour is not the Orbbec's, so **do not mix the two sets in one dataset** (use a new `--root`).
Not yet tuned or run on the real camera: exposure / white balance are the Xtion's own auto. `settings:` accepts
`auto_exposure`, `exposure`, `gain`, `auto_white_balance` and the software `rgb_gain`; a value the camera refuses is an error.
Aim and check the cameras live (in a browser, with fps / brightness / blown-out % on each tile):

    .venv/bin/python tools/camera_preview.py --cameras xtion_top [--grid] [--port 8790]      # then open http://localhost:8790

It opens the cameras exactly as the recorder does, so stop it (Ctrl-C) before recording: a camera has one owner at a time.

### Exposure and colour

Per camera, `settings:` in the config. Every start begins from a defined state (auto exposure, auto white balance, constant
frame rate; manual values persist inside the camera otherwise) and then applies what you list. A value the camera refuses is an
error, never silently ignored.

| setting | meaning |
|---|---|
| `exposure` | exposure time in units of 100 us (200 = 20 ms); switches auto exposure off. 200 is a multiple of the 10 ms mains-flicker period (no banding); at 30 fps the longest usable value is ~330 |
| `gain` | 0..100, brightens at the cost of noise |
| `white_balance` | fixed colour temperature, 2800..6500 K; switches auto white balance off |
| `rgb_gain` | `[r, g, b]` software multipliers on every frame: a constant white balance on top of the camera's (see below) |
| `auto_exposure`, `auto_white_balance`, `ae_priority`, `hue` | the camera's own switches |

**Why fixed exposure.** Auto exposure was unstable: the same scene came out at brightness 112 in one run and 188-217 with 30-45 %
of the pixels blown out in the next, and it shifts as the arm moves, which is brightness noise in the dataset. The shipped
config fixes exposure 200 with gain 40 (`cam_high`) / 50 (`cam_left_wrist`): no blown highlights, the work surface well exposed.
**Re-tune whenever the lighting changes** (windows, lamps):

    .venv/bin/python tools/camera_tune.py --serial CC1365303LL --out logs/tune          # scores a sweep + contact sheet

**White balance.** Auto white balance assumes the scene averages to grey, so a large coloured object (the green one) pulls it
toward magenta. The camera's fixed values have no neutral point either. For true neutral calibrate on a white sheet: hold plain
white paper evenly lit over the middle of the view, about 20 cm from the lens (use the exposure/gain you record with):

    .venv/bin/python tools/camera_tune.py --serial CC1WC5200WW --out logs/card --white-card --exposure 200 --gain 50

It prints the `rgb_gain: [...]` line for `config/teleop.yaml` and saves before/after pictures. Do each camera once per lighting.

Nothing else is graded: frames are stored as the camera delivers them (plus the constant gain above). A fancier grade would
have to be applied identically when the trained policy runs, so it stays off. Each session's camera settings are saved in
`meta/recording_setup.json` inside the dataset so inference can be given matching images. Video is h264 at `crf: 20`.

## Formats

Both layouts were checked against LeRobot-written datasets, and loaded and decoded with the real loaders: v3.0 with
LeRobot 0.5.1, v2.1 with OpenPI's pinned LeRobot.

- **v3.0**: `data/chunk-000/file-000.parquet`, `videos/<key>/chunk-000/file-000.mp4` (episodes concatenated; rolls to a new file at
  100 MB / 200 MB), `meta/{info.json, stats.json, tasks.parquet, episodes/...}`.
- **v2.1** (OpenPI): one parquet and one mp4 per episode, `meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl}`.

Video is h264 (decodes everywhere), keyframe every 2 frames like LeRobot. An episode is fully written before the metadata that
names it, so an interrupted save leaves a dataset that still loads.

## Check and use it

    .venv/bin/python tools/validate_dataset.py datasets/pick_cube             # structure: rows, video frames, timestamps
    .venv/bin/python tools/validate_dataset.py datasets/pick_cube --lerobot   # also LeRobotDataset + decode (needs lerobot)

Push the directory as a Hugging Face dataset repo (LeRobot also looks for a git tag named after the codebase version, `v3.0` /
`v2.1`) or just copy it to the training machine. For OpenPI record with `--format v2.1`.

## Not there yet

- Only arm-to-arm teleop is wired in. Quest teleop can plug into the same `RecordingSession` (it only needs a
  `(state, action)` pair per tick, see `ArmToArmSource` in `piper_teleop/record.py`).
- No audio, no depth stream (colour only), no joint velocities/effort.
- Existing datasets can't be converted between v2.1 and v3.0 here; record in the one you need.
