# TAKOS

MuJoCo + ROS 2 simulation for a tendon-driven tentacle arm.

## Quick setup

```bash
# from repo root
python3 -m venv venv
source venv/bin/activate
pip install -r src/requirements.txt
source /opt/ros/humble/setup.bash
```

## Run simulation

```bash
cd src
python3 run_sim.py                  # defaults to tentacle_locked.xml
# or
python3 run_sim.py tentacle_locked_tipfirst.xml
```

`run_sim.py` now prints at 0.1s intervals:
- mode
- tip position `(x, y, z)`
- tendon lengths `(L1, L2)`

## Send commands

```bash
# tendon mode: [act1, act2] in [0, 1]
ros2 topic pub --once /tentacle/tendon_cmds std_msgs/msg/Float64MultiArray "{data: [0.5, 0.2]}"

# joint mode: 23 values in [-1, 1]
ros2 topic pub --once /tentacle/tendon_cmds std_msgs/msg/Float64MultiArray "{data: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}"
```

## Generate `(x, y) -> string length` dataset

```bash
cd src
python3 collect_xy_string_dataset.py \
  --xml tentacle_locked_tipfirst.xml \
  --sample-dt 0.1 \
  --target-unit cm \
  --length-unit cm \
  --output xy_string_dataset_tipfirst.csv \
  --metrics-output xy_target_metrics_tipfirst.csv \
  --verification-output xy_model_verification_tipfirst.json
```

Output format:

```text
target_x,target_y,orig_string_l,orig_string_r,new_string_l,new_string_r
```

## Core files

- `src/run_sim.py` - MuJoCo loop + ROS topics
- `src/tentacle_locked.xml` - baseline model
- `src/tentacle_locked_tipfirst.xml` - tuned tip-first model
- `src/collect_xy_string_dataset.py` - dataset + verification generator
