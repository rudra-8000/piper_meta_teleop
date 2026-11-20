# Driving the robot from Python: a trained policy, a controller or a planner

`piper_teleop/robot.py` is the importable control layer; `piper_teleop/run_policy.py` is a ready-made loop around it. No ROS, no
Quest, no leader arm. It drives ONE arm (by default the physical **left** arm, the one the datasets describe).

    from piper_teleop.robot import PiperRobot, Rate

    with PiperRobot(arm="left", cameras="xtion_top") as robot:      # sim=True: simulated arm, no hardware
        robot.go_rest()                                                # fold to the rest pose (where every recorded episode starts)
        rate = Rate(30)
        for _ in range(900):
            obs = robot.get_observation()                            # same keys / units as the recorded dataset
            action = my_policy(obs)                                  # (7,) = 6 joints [rad] + gripper 0..1
            robot.send_action(action)
            rate.sleep()

## Observation and action = the dataset's format

| | |
|---|---|
| `obs["observation.state"]` | (7,) float32: follower joints 1..6 [rad], gripper opening 0..1 (1 = open, 70 mm) |
| `obs["observation.images.<cam>"]` | (H, W, 3) uint8 RGB, one per camera of `record.cameras` (or the `camera_sets` entry you pass) |
| action | (7,) absolute joint targets [rad] + gripper target 0..1: what the dataset's `action` holds |

A policy trained on a dataset from `piper_teleop.record` therefore needs no conversion. Use the **same camera set** you recorded
with (`cameras="xtion_top"` or the default), at the same fps (30), or the images will not look like the training data.

## API (all in `piper_teleop.robot`)

| | |
|---|---|
| `PiperRobot(arm, sim=, cameras=, max_joint_speed=, ...)` | `cameras`: None = config `record.cameras` (none in sim), `"xtion_top"` = a `record.camera_sets` entry, a dict, `False` = none, `"fake"` = test pattern |
| `connect()` / `with ... as robot` / `close()` | enables the arm, starts the control thread and the cameras |
| `get_observation()` | the latest state + frames (a dict, see above) |
| `state`, `tcp_pose()`, `commanded_q` | measured (7,) state; tool tip `(R, p)` in the arm's base frame (x forward, y left, z up) |
| `send_action(action, speed=None)` | set the target; returns at once, the arm moves there at the speed cap |
| `run_chunk(actions, hz=30)` | execute an (N, 7) chunk open-loop, blocking |
| `send_tcp_pose(p, rotation=None, gripper=None)` | tool-tip target via IK, for planners. `rotation`: 3x3 / quaternion xyzw / `None` keep current / `"free"` position only |
| `go_to(q)`, `go_rest()`, `hold()`, `cancel_moves()` | blocking joint-space move, fold to rest, stop where it is, abort a move in progress |
| `Rate(hz)` | `rate.sleep()` holds a loop at a fixed rate and never bursts to catch up |

How it moves: a background thread streams to the arm at 200 Hz (`robot.control_hz`), stepping the commanded joints toward the
latest target at most `max_joint_speed` (2 rad/s by default, all joints arrive together), through the same gripper layer as the
teleop (the Piper gripper only follows a slowly varying command). So call `send_action` at your policy's rate; between calls the
arm keeps moving smoothly, and if you stop calling it the arm holds. The **first** target after connecting is approached at the
slow `sync_speed` (0.5 rad/s), like the teleop's start-up alignment.

`send_tcp_pose` refuses instead of sending something different: `ValueError` outside the workspace box
(`defaults.workspace`, `arms.left.workspace`), `IKError` if the pose is not reached to 4 mm / 5 deg, or if the solution needs a joint
to swing more than 57 deg from where the arm is (a different arm configuration: go via intermediate waypoints). Holding the rest
orientation while moving sideways close to the base is exactly such a case, so use `rotation="free"` for position-only moves.
`examples/cartesian_demo.py` draws a square that way.

Want a separate process or machine instead (your own client, any language)? See [remote_control.md](remote_control.md): a server wraps this
same class, and `PiperClient` has the same methods.

## Run a policy with the script

    python -m piper_teleop.run_policy --policy MODULE:FACTORY [--arg k=v] --prompt "clear the table" --cameras xtion_top [--hz 30] [--sim]

`--policy` is `package.module:factory` or `path/to/file.py:factory`. The factory gets the `--arg`s and returns
`policy(obs) -> (7,) action or (N, 7) chunk`; `obs` is `get_observation()` plus `obs["prompt"]`. Chunks run open-loop, `--execute N` actions
before the policy is asked again. The script confirms, folds the arm to rest, runs at `--hz`, and stops on Ctrl-C / `--steps` /
`--duration-s`: the arm holds, then folds to rest. A **second Ctrl-C** skips the folding and just leaves it holding. Examples in
`examples/policies.py`:

    python -m piper_teleop.run_policy --policy examples/policies.py:hold --sim              # does nothing: checks the plumbing
    python -m piper_teleop.run_policy --policy examples/policies.py:sine --sim --steps 300  # gentle J2/J3 swing + gripper
    python -m piper_teleop.run_policy --policy examples/policies.py:replay --arg root=datasets/clear_table --arg episode=3 --yes
                                   # replays a recorded episode's actions: the best first real-arm test of the action path

## Plugging in a model

A LeRobot policy (ACT, Diffusion, ...) in ten lines, **not tested here** (no policy has been trained on this setup yet):

    # my_policy.py
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy          # import path depends on your lerobot version

    def make(path):
        policy = ACTPolicy.from_pretrained(path).to("cuda").eval()
        def act(obs):
            batch = {"observation.state": torch.from_numpy(obs["observation.state"])[None].cuda()}
            for k, v in obs.items():
                if k.startswith("observation.images."):
                    batch[k] = torch.from_numpy(v).permute(2, 0, 1)[None].float().div(255).cuda()
            with torch.no_grad():
                return policy.select_action(batch).cpu().numpy()[0]  # one (7,) action per call
        return act

    python -m piper_teleop.run_policy --policy my_policy.py:make --arg path=outputs/act_clear_table --cameras xtion_top

OpenPI: run its policy server and call it from the factory (`openpi_client`'s `WebsocketClientPolicy(host, port).infer(...)["actions"]`
returns an (N, 7) chunk); build the observation dict the way your OpenPI data config expects. A high-level planner or LLM agent is
the same idea: any callable that turns `obs` into an action, or skip the script and call `send_tcp_pose` yourself.

## Safety, and what has not been tested

* Joints are clipped to the arm's limits and the speed is capped; the joint lock (`config/joint_lock.yaml`) still applies.
* The motors are **never** disabled: on exit the arm folds to rest (slowly) and stays enabled and holding. If the control thread
  dies (CAN error) the next call raises and the arm holds its last command.
* A policy that outputs absolute targets far from the arm is rate-limited, but it will still move the arm at up to
  `max_joint_speed` along a straight joint-space line. Start with `--max-joint-speed 0.5`, the `hold` / `replay` policies and a clear table.
* Settings: `robot:` in `config/teleop.yaml` (arm, speeds, rest pose, park on exit).
* Everything here is tested in simulation only (tests/test_robot.py). It has **not** been run on the real arm.
