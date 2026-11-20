# Piper teleop (no ROS)

Teleoperation for two AgileX Piper arms, plain Python + CAN:

| Mode | Command |
|------|---------|
| Quest 3 → both arms (headset on, or headless / controllers only) | `python -m piper_teleop.quest_server` |
| Leader arm → follower arm (software) | `python -m piper_teleop.leader_follower` |
| Record a LeRobot dataset (v3.0 / v2.1, 30 Hz) while teleoperating arm-to-arm | `python -m piper_teleop.record --root datasets/NAME` - see [docs/recording.md](docs/recording.md) |
| Same, no hardware (simulated arms) | add `--sim` |

Read the two PDFs in `docs/` first:

* `docs/Piper_Setup_Guide.pdf` – unboxing, power, CAN adapters, first-contact tests
* `docs/Teleop_User_Guide.pdf` – every mode, controls, latency tuning, safety

Quick start: `scripts/install.sh`, then `python -m piper_teleop.quest_server --sim` and open
<http://localhost:8788/dashboard>. Tests: `.venv/bin/python -m pytest -q tests`.
Rebuild the PDFs: `.venv/bin/python docs/build_pdfs.py`. Settings: `config/teleop.yaml`.

**Driver:** AgileX's current library [pyAgxArm](https://github.com/agilexrobotics/pyAgxArm) by default
(`piper.driver: pyagxarm`); the older `piper_sdk` is kept as a fallback and for the diagnostic tools.
**Joint lock:** `config/joint_lock.yaml` can make an arm gripper-only until a given time (enforced at CAN level).
**Camera:** ASUS Xtion depth camera - `.venv_cam/bin/python tools/xtion_snap.py OUT_DIR` (Python 3.10 env, OpenNI2).

**Status:** tested in simulation and against the SDK's kinematics; not yet run on real arms or a real
Quest. See chapter 10 of the user guide.
