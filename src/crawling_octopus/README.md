# Octopus arm: simulation, training and deployment toolkit

MuJoCo + PPO tools for a tendon-driven spiral ("octopus") arm: train a single arm to crawl, train a four-arm
octopus to walk, move the arm tip to chosen positions, check CAD meshes, and run a trained policy on a Raspberry Pi.

Everything here was written in conversation and is split into two folders that can live in separate directories:

```
octopus_crawl/     single arm that crawls (2 muscles + 23 joint actuators in the simulation)
octopus_walker/    four arms x 3 tendons: model generator, walking training, reaching, helpers
```

Both folders need a link mesh (`baselink.stl`, or your own Fusion export) next to whichever XML you load,
because the XML finds meshes in its own folder (`meshdir="."`).

---

## 1. Status: what has actually been run

| Part | Status |
|---|---|
| `patch_model.py`, `check_model.py`, `diagnose.py`, `train.py`, `evaluate.py` (single arm) | Run on your machine, with the output you pasted |
| `gait_test.py`, `debug_env.py`, `export_policy.py` | Written and compile-checked; no results reported yet |
| `pi_runner.py` | Dry-run logic tested on a synthetic policy (numpy only) |
| `check_stl.py` | Tested on synthetic good / inverted / holed / off-centre meshes |
| `make_octopus_xml.py` | Output checked for well-formed XML, unique names, valid tendon references and tendon lengths; never loaded in MuJoCo by the author |
| `reach_ik.py` | Solver core tested on a synthetic arm (median error about 1 mm); the MuJoCo data collection is untested |
| `octopus_walk_env.py`, `train_walk.py`, `evaluate_walk.py`, `walk_gait_test.py` | Written and compile-checked; **not run** |

Nothing was run against MuJoCo by the author, so treat every "not run" item as needing a first test from you.

---

## 2. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # mujoco, gymnasium, stable-baselines3, numpy, tensorboard, imageio[ffmpeg]
```

- Run scripts from the folder that contains the XML, or pass `--xml path`.
- **macOS:** the live viewer (`--viewer`) must be started with `mjpython` instead of `python`. Videos use plain `python` and need no `MUJOCO_GL` setting. On headless Linux, set `MUJOCO_GL=egl` for videos.
- Training runs on the CPU (`--device cpu`) and uses one process per environment (`--n-envs`, default 8).

---

## 3. Workflows

### A. Single arm crawls (`octopus_crawl/`)

```bash
# 1. build the simulation model from the original XML (original is never modified)
python patch_model.py --xml spiral_robot.xml --base-mass 0.5          # -> spiral_robot_v2.xml
# 2. check it
python check_model.py --xml spiral_robot_v2.xml                       # mass, resting pose, muscle bend, speed
python diagnose.py   --xml spiral_robot_v2.xml                        # contacts, torque budget, motion tests
python gait_test.py  --xml spiral_robot_v2.xml --masses 0.12,0.5      # can ANY open-loop gait move it?
# 3. train (smoke test first, then the full run)
python train.py --xml spiral_robot_v2.xml --run-name smoke --timesteps 100000
python train.py --xml spiral_robot_v2.xml --run-name crawl_v1
tensorboard --logdir runs
# 4. look at it
python evaluate.py --run runs/crawl_v1 --episodes 5
python evaluate.py --run runs/crawl_v1 --video crawl.mp4
mjpython evaluate.py --run runs/crawl_v1 --viewer
```

With the motor speed limit (40 rpm) use slower gaits and longer episodes:

```bash
python train.py --xml spiral_robot_v2.xml --run-name slow_v1 \
  --servo-rpm 40 --spool-radius 0.010 --phase-period 4 --episode-seconds 20
```

If traction is the problem (heavy base, light tip), give the arm grip and the base slip:

```bash
python patch_model.py --xml spiral_robot.xml --base-mass 0.5 \
  --ground-friction 0.2 --arm-friction 2.0 --base-friction 0.2
```

### B. Four-arm octopus walks (`octopus_walker/`)

```bash
# 1. the model: generate one, or use your own (Fusion-scaled) octopus.xml
python make_octopus_xml.py                                   # 4 arms, 3 tendons each -> octopus.xml
python alternate_axes.py --xml octopus.xml --out octopus_alt.xml   # optional: halves the joints, faster sim
# 2. checks BEFORE training
python walk_gait_test.py --xml octopus.xml --mode bend                 # is the muscle force sensible?
python walk_gait_test.py --xml octopus.xml --mode gait --force 8       # can any hand-made gait move the body?
# 3. train
python train_walk.py --xml octopus.xml --run-name smoke --timesteps 100000 --servo-rpm 40 --spool-radius 0.010
python train_walk.py --xml octopus.xml --run-name walk_v1 --timesteps 10000000 --servo-rpm 40 --spool-radius 0.010
# 4. evaluate
python evaluate_walk.py --run runs/walk_v1 --episodes 5
python evaluate_walk.py --run runs/walk_v1 --video walk.mp4
```

If you change the muscle force, edit `force="..."` on the muscles in the XML (or regenerate with `--force`): the
`--force` flag of `walk_gait_test.py` only changes the test.

### C. Move the tip to a chosen position (`octopus_walker/reach_ik.py`)

```bash
python make_octopus_xml.py --arms 1 --rig --alternate-axes --out segment_test.xml   # one arm, base fixed in the air
python reach_ik.py collect   --xml segment_test.xml --n 2000 --out reach_data.npz
python reach_ik.py workspace --data reach_data.npz                   # what can it reach?
python reach_ik.py solve     --data reach_data.npz --target 0.30 0.05 -0.10 --xml segment_test.xml
python reach_ik.py add       --data reach_data.npz --u 0.2 0.5 0.1 --tip 0.31 0.04 -0.12   # real measurement
```

### D. Run a trained crawl policy on a Raspberry Pi (`octopus_crawl/`)

```bash
python export_policy.py --run runs/crawl_v1                          # -> runs/crawl_v1/policy.npz, obs_log.npz
scp runs/crawl_v1/policy.npz runs/crawl_v1/obs_log.npz pi_runner.py pi@raspberrypi.local:~/octopus/
python3 pi_runner.py --policy policy.npz --dry-run obs_log.npz       # on the Pi: timing + correctness, no hardware
```

The Pi needs only `numpy`. Hardware mode is deliberately not implemented, see Limitations.

### E. CAD meshes (`octopus_crawl/check_stl.py`, `octopus_walker/arm_dimensions.csv`)

```bash
python check_stl.py my_link.stl       # units, origin/axis, watertight, inverted faces, mass
```

`arm_dimensions.csv` lists per-link dimensions in millimetres (scale, pivot heights, tendon-channel radii and
heights) derived from the simulation numbers, for modelling the links in CAD.

---

## 4. File reference

### `octopus_crawl/` (single arm)

| File | What it does |
|---|---|
| `octopus_env.py` | Gymnasium environment for crawling. See section 5 for actions, observations and reward |
| `train.py` | PPO training. Writes `runs/<name>/` |
| `evaluate.py` | Runs a trained policy; prints return, distance, speed, lift, joint speed, bend; `--video` or `--viewer` |
| `patch_model.py` | Writes `<xml>_v2.xml`: density, muscle force, joint softness, base mass, friction layout, link-link collision off |
| `check_model.py` | Pre-flight: counts, mass, rest pose, per-tendon muscle bend (joint actuators switched off), physics speed |
| `diagnose.py` | Contacts at rest, per-joint torque budget (weight vs muscle), motion tests |
| `gait_test.py` | Open-loop muscle sine waves; reports the best distance for light/heavy base and two friction layouts |
| `debug_env.py` | Random-action vs trained-policy test; prints actions, NaN flags and MuJoCo warnings (finds frozen policies) |
| `export_policy.py` | Exports the actor network + normalisation to `policy.npz`, records `obs_log.npz`, verifies the numpy copy |
| `pi_runner.py` | numpy-only policy runner for the Pi; dry-run replay; hardware stubs |
| `check_stl.py` | Sanity-checks an STL exported from CAD |
| `requirements.txt` | Python dependencies |

### `octopus_walker/` (octopus)

| File | What it does |
|---|---|
| `make_octopus_xml.py` | Generates the octopus MJCF (see options below) |
| `octopus.xml`, `segment_test.xml` | Generated examples (4 arms x 3 tendons; one hanging arm). Your Fusion-scaled `octopus.xml` supersedes these |
| `alternate_axes.py` | Converts two hinges per link into alternating single hinges (about half the joints); keeps header comments |
| `octopus_walk_env.py` | Walking environment for the tendon-only octopus (12 actions) |
| `train_walk.py` / `evaluate_walk.py` | PPO training and evaluation for walking |
| `walk_gait_test.py` | `--mode bend`: how far one arm curls per command level. `--mode gait`: 36 open-loop gaits |
| `reach_ik.py` | Data-driven inverse kinematics: `collect`, `workspace`, `solve`, `add` |
| `arm_dimensions.csv` | Per-link CAD dimensions in mm |

### `make_octopus_xml.py` options

| Option | Default | Meaning |
|---|---|---|
| `--arms`, `--mount-angles` | 4, 45/135/225/315 | Number of arms and their directions (degrees, counter-clockwise from +x) |
| `--tendons` | 3 | Tendons per arm, equally spaced around the axis |
| `--bend-axes`, `--alternate-axes` | 2 | Two hinges per link, or one hinge alternating x/y |
| `--segments` | 1 | Independently actuated segments per arm (tendons anchor at each segment's end link) |
| `--rig`, `--rig-height` | off, 0.35 | Fix the torso in the air (test rig for one arm) |
| `--body-radius`, `--body-half-height`, `--body-mass` | 0.12, 0.045, 1.5 | Torso cylinder and its mass |
| `--density`, `--force` | 1100, 30 | Arm material density; muscle peak force (N) |
| `--stiffness-scale`, `--damping-scale`, `--frictionloss-scale` | 0.3, 0.5, 0.3 | Passive joint properties |
| `--yaw`, `--mount-mass`, `--yaw-*` | off | Optional yaw servo at each arm root |
| `--joint-actuators` | off | Simulation-only joint position actuators (not realistic for the hardware) |

---

## 5. Concepts

### Outputs of a training run: `runs/<run-name>/`
`config.json` (all settings, so evaluation rebuilds the same environment), `final_model.zip`, `vecnormalize.pkl`
(observation statistics, **must be paired with the model**), `checkpoints/` (every 200k steps for the crawler,
500k for the walker; each `.zip` has a matching `..._vecnormalize_<N>_steps.pkl`).
TensorBoard logs go to `runs/` as well.

### Single-arm environment (`octopus_env.py`)
- **Actions** (`--action-mode reduced`, default): 2 muscle commands plus 4 spatial-mode coefficients that set all 23 joint
  targets (cosine modes along the arm). `--action-mode full`: 2 muscles plus 23 independent joint targets.
- **Observation** (61 + action size values; 67 in the default mode): joint angles and velocities, gravity direction and
  angular velocity of the base, centre-of-mass velocity, 2 tendon lengths, 2 muscle activations, previous action, gait clock.
- **Time:** 20 ms control step (20 x 1 ms physics), 8 s episodes (400 steps) by default. An iteration is 8 envs x 512 steps
  = 4096 timesteps, so 3M timesteps is about 733 iterations.
- **Termination:** base rolls over (upright value below 0.3) or the centre of mass rises above 0.5 m; otherwise truncated at the time limit.
- **Reward** (per control step; override with `--weight NAME=VALUE`):

| Weight | Default | Rewards or penalises |
|---|---|---|
| `vel` | 10 | + forward speed along +x (clipped to 0.5 m/s) |
| `lateral`, `roll`, `yaw` | 5, 0.5, 0.5 | Sideways speed, base tipping, heading drift |
| `action_rate`, `energy` | 0.5, 0.02 | Jerky commands, muscle effort |
| `height`, `tip_lift`, `vert_vel` | 30, 10, 10 | Links lifted above rest height, tip lift, fast up/down motion |
| `joint_vel`, `joint_acc`, `bend` | 0.05, 0.02, 0.5 | Fast joints, thrashing, large joint angles |
| `fall` | 5 | One-off penalty when the base tips over |

- `evaluate.py` reports: **distance** (m moved along +x), **mean speed**, **mean lift** (cm), **joint speed** (RMS rad/s), **bend** (mean squared joint angle as a fraction of the limit).

### Walking environment (`octopus_walk_env.py`)
- **Actions:** 12 values in [-1, 1] mapped to muscle commands in [0, 1]; order `a1_m0, a1_m1, a1_m2, a2_m0, ...`.
- **Observation (32 values):** gravity direction in the body frame, gyro, 12 tendon lengths (encoder equivalent), 12 commanded values, gait clock.
- **"Forward"** is the torso's own +x axis, so the policy never needs the world heading.
- **Reward:** `vel` 30 (speed along torso +x, clipped to 0.3 m/s), `lateral` 5, `tilt` 1, `height` 0 (set 20 to encourage lifting the body), `action_rate` 0.05, `energy` 0.02, `fall` 5.
- **Robustness:** sensor noise and gyro bias (`--noise`), one control step of latency (`--latency`), random friction (±40%), torso mass (±20%) and starting heading (`--no-randomize` turns the randomisation off).

### Servo speed limit
Tendon speed = motor rad/s x spool radius. 40 rpm is 4.19 rad/s, so a 10 mm spool pulls 4.2 cm/s. With `--servo-rpm`, `--spool-radius`
and `--full-stroke` (tendon travel for command 0 to 1; 0.19 m for the original arm, 0.08 m for your rescaled arm), the muscle commands
cannot change faster than the motor can move the tendon. Use a longer gait clock (`--phase-period`) and longer episodes when you enable it.

### Naming and frames in the octopus XML
`a<i>_link<k>` bodies, `a<i>_s<k>_t<j>a/b` tendon sites, `a<i>_t<j>` tendons, `a<i>_m<j>` muscles, `a<i>_jx<n>/jy<n>` hinges,
`torso`, free joint `root`, sensors `body_quat`, `body_gyro`, `body_acc`, `a<i>_t<j>_len`. Arm axis = local +z, "up" = local +y; tendon 0
is the up side, tendons 1 and 2 follow at 120 degree steps.

---

## 6. Troubleshooting

| Symptom | Likely cause | What to try |
|---|---|---|
| Arm only twitches, barely moves forward | Penalty weights outweigh the forward reward | Lower `bend`, `joint_acc`, `joint_vel`, `height`; raise `vel`; do a sanity run with all new penalties set to 0 |
| Arm does not move at all | Traction/mass problem, or a frozen policy | `gait_test.py` (physics), `debug_env.py` (look for `BADCTRL` / NaN), friction flags in `patch_model.py` |
| Motion is violent | Weak smoothness penalties | Raise `joint_vel`, `joint_acc`, `vert_vel`; lower `--smooth`; enable the servo limit |
| Muscles slam every joint to its limit | Muscle force too high for the arm size | `check_model.py` / `walk_gait_test.py --mode bend`; lower `force` |
| Muscle test shows "barely moves" | Joint position actuators fighting the muscles | They are switched off in the tests; trust the per-joint angles, not a mean |
| MuJoCo cannot load the XML or mesh | Inverted/non-watertight STL, wrong units or origin | `check_stl.py` on the STL; check `scale` in the `<mesh>` tag |
| Evaluation gives distance 0.000 and zero joint speed | Old policy on a changed model, or NaN controls | Retrain; `debug_env.py` |
| Training is slow | Many hinges, few CPU cores | `alternate_axes.py`, `--frame-skip 10`, `--n-envs` = your core count |
| Policy behaves randomly after loading | Model and `vecnormalize` file from different runs/checkpoints | Pair `ppo_<N>_steps.zip` with `ppo_vecnormalize_<N>_steps.pkl` |

Tips: judge a policy by the evaluation **numbers and the video**, not the reward alone; a policy can score well by flipping or shaking.
Policies trained on an older model or reward are not valid after you change the XML or the reward: retrain.

---

## 7. Limitations and not-yet-done

- **The crawl policy cannot drive the real arm.** It observes joint angles and centre-of-mass velocity and commands 23 joint actuators that a tendon-only arm does not have. A hardware-matched (tendon-only, encoder + IMU) single-arm environment has not been written. The walking environment is built that way.
- **`pi_runner.py` and `export_policy.py` target the crawl policy** (they import `OctopusArmEnv`; `pi_runner.py`'s command mapping assumes 2 muscles + 23 joints). They do not handle the walking policy yet. Hardware mode (`read_observation`, `send_to_servos`) is intentionally unimplemented because it depends on your servos and sensors.
- **`train.py` resume does not reload the observation statistics.** Use `train_walk.py` (it does, with `--resume-stats`) or avoid resuming the crawler.
- **Link-to-link collision is off**, so arms can pass through the torso and each other; check the video for it.
- **Friction is isotropic and uncalibrated.** Real-arm friction, tendon slack and servo lag are not modelled beyond the rate limit and noise.
- **Muscle length ranges for new geometry are scaled approximations** from the original arm.
- **The octopus XML generator makes the original-scale model.** Your Fusion-scaled `octopus.xml` was edited separately (the header comments record what changed); regenerating to the same filename would overwrite it.
- **Reaching** with one 3-tendon segment reaches a 2-D surface; use `--segments 2` or more for a 3-D volume.
- **Walking is a hard learning problem.** Expect 10M or more steps, and run the open-loop gait test first.

## 8. Licence note
If you use the Open-Spiral-Robots repository (`ZhanchiWang/Open-Spiral-Robots`) for the arm design or exports, it is licensed
PolyForm Noncommercial 1.0.0 and asks to be cited in academic work.