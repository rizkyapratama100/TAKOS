# Octopus arm crawling (MuJoCo + PPO)

## Layout

Put these files in one folder together with **`spiral_robot.xml`** and **`baselink.stl`**
(the XML loads the mesh from its own directory via `meshdir="."`).

```
octopus_env.py    Gymnasium env (obs, actions, reward)
train.py          PPO training (stable-baselines3, parallel envs, VecNormalize, checkpoints)
evaluate.py       Run a policy, print distance/speed, record mp4 or open the viewer
check_model.py    Pre-flight check of the MJCF (mass, rest pose, can muscles bend the arm?)
requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python check_model.py --xml spiral_robot.xml          # do this first
python train.py --xml spiral_robot.xml --timesteps 3000000 --n-envs 8 --run-name crawl_v1
tensorboard --logdir runs
python evaluate.py --run runs/crawl_v1 --video crawl.mp4   # headless: MUJOCO_GL=egl
```

## How the model is used

* The arm lies along world +x; its hinges bend it in the vertical x-z plane. Tendon 1 is the top
  muscle, tendon 2 the bottom muscle. Crawling is therefore planar (inchworm / arch-and-push),
  and the base can also roll about the arm axis, which is penalised and ends the episode.
* Actions (25 actuators in the XML):
  * `reduced` (default): 2 muscle activations + 4 spatial-mode coefficients (cosine modes along the
    arm) that set all 23 joint position targets. Much easier for PPO than 25 independent outputs.
  * `full`: 2 muscles + 23 independent joint targets (`--action-mode full`).
  * Commands are low-pass filtered (`--smooth`) and the policy sees a 1 s sin/cos clock to make
    rhythmic gaits easy to find.
* Reward: forward COM speed along +x (clipped at 0.5 m/s) minus sideways speed, roll/yaw drift,
  action jerk and effort. Change direction or weights via `OctopusArmEnv(direction=..., weights=...)`.
* Observation: joint angles/velocities, gravity in base frame, base angular velocity, COM velocity,
  normalised tendon lengths, muscle activations, previous action, clock.

## Caveats and tuning

* I could not run any of this here (no `baselink.stl`, no MuJoCo install), so treat the first
  `check_model.py` run as the real test.
* The XML has density 5000 and muscles of only 10 N. If `check_model.py` says full muscle
  activation barely bends the arm, raise the muscle `force` (or lower `density` / joint `stiffness`)
  before training; no reward shaping will fix an arm that cannot move.
* Friction is isotropic (`1 0.5 0.5`). Crawling then relies on lifting and sliding parts of the arm
  at different times. If learning stalls, try higher ground friction, e.g. `friction="1.5 0.5 0.5"`
  on the ground geom, and more `--timesteps`.
* Meshes collide as convex hulls, so non-adjacent links may touch. If the rest pose looks jittery
  (`check_model.py` prints contact count), add `<contact><exclude .../></contact>` pairs.
* Slow sim? Use `--frame-skip 10` and shorter `--episode-seconds`.
* The first thing to look at in `evaluate.py --video` is whether the policy discovers a gait or
  exploits something (flipping, launching). Adjust the termination and penalty terms accordingly.
