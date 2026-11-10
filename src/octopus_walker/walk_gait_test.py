"""Two quick checks to run BEFORE training the octopus to walk.  No learning involved.

1) bend check: how far does ONE arm curl for a given command on one tendon?  (is the muscle force sensible?)
       python walk_gait_test.py --xml octopus.xml --mode bend
   If a command of 0.25 already curls the arm to its joint limits, the force is too high for the arm's size:
   the policy would only have a tiny usable range. Lower it with --force (try 6-10 N for a 19 cm arm).

2) gait test: drive all 12 tendons with open-loop rotating-bend patterns (each arm's tip circles) and measure how
   far the body moves.  If no pattern moves the body, check friction/mass/force before training.
       python walk_gait_test.py --xml octopus.xml --mode gait --force 8
   Patterns: sync (all arms together), diagonal (opposite arms in antiphase), wave (arms 90 deg apart).
   Frequencies are low because the real servos are slow.
"""
import argparse
import itertools
import math

import mujoco
import numpy as np


def load(xml, force):
    m = mujoco.MjModel.from_xml_path(xml)
    if force is not None:
        m.actuator_gainprm[:, 2] = force      # muscle peak force lives in gainprm[2] and biasprm[2]
        m.actuator_biasprm[:, 2] = force
    return m, mujoco.MjData(m)


def arm_actuators(m):
    n = m.nu // 3
    return {i: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a{i}_m{j}") for j in range(3)]
            for i in range(1, n + 1)}


def settle(m, d, steps=500):
    for _ in range(steps):
        mujoco.mj_step(m, d)


def bend_check(args):
    m, d = load(args.xml, args.force)
    mu = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "a1_m0")
    joints = [j for j in range(m.njnt) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("a1_j")]
    adr = [m.jnt_qposadr[j] for j in joints]
    print(f"arm 1: {len(joints)} bending hinges, muscle force {m.actuator_gainprm[mu, 2]:g} N")
    print("command on tendon a1_m0 only -> total curl (sum |angle|), hinges at their limit, tip height gain")
    tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "a1_link24")
    for level in (0.1, 0.25, 0.5, 0.75, 1.0):
        mujoco.mj_resetData(m, d)
        d.ctrl[:] = 0.0
        settle(m, d, 300)
        z0 = d.xpos[tip][2]
        d.ctrl[mu] = level
        settle(m, d, 3000)
        q = d.qpos[adr]
        at_limit = int((np.abs(q) > 0.95 * 0.5236).sum())
        print(f"  u={level:4.2f}: total curl {np.abs(q).sum():5.2f} rad, {at_limit:2d}/{len(q)} hinges at limit, "
              f"tip rose {(d.xpos[tip][2] - z0) * 100:+5.1f} cm")
    print("A good setting: u=1.0 reaches (nearly) the limits but u=0.25-0.5 is clearly partial.")


def run_gait(m, d, freq, amp, pattern, direction, seconds):
    A = arm_actuators(m)
    mujoco.mj_resetData(m, d)
    d.ctrl[:] = 0.0
    settle(m, d, 500)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    com0 = d.subtree_com[torso][:2].copy()
    up_min, dt = 1.0, m.opt.timestep
    for k in range(int(seconds / dt)):
        t = k * dt
        for i, ids in A.items():
            theta = 0.0 if pattern == "sync" else (math.pi * ((i - 1) % 2) if pattern == "diagonal" else (i - 1) * math.pi / 2)
            psi = 2.0 * math.pi * freq * t * direction + theta
            for j in range(3):
                alpha = math.radians(90 + 120 * j)         # tendon 0 is the "up" side of the arm
                d.ctrl[ids[j]] = min(max(0.5 + amp * math.cos(psi - alpha), 0.0), 1.0)
        mujoco.mj_step(m, d)
        if k % 50 == 0:
            up_min = min(up_min, float(d.xmat[torso].reshape(3, 3)[2, 2]))
    return d.subtree_com[torso][:2] - com0, up_min


def gait_test(args):
    m, d = load(args.xml, args.force)
    results = []
    for freq, amp, pattern, direction in itertools.product(
            (0.1, 0.2, 0.4), (0.5, 0.8), ("sync", "diagonal", "wave"), (1, -1)):
        disp, up = run_gait(m, d, freq, amp, pattern, direction, args.seconds)
        results.append((float(np.linalg.norm(disp)), disp, up, freq, amp, pattern, direction))
        print(f"  {pattern:8s} f={freq:.1f} Hz amp={amp:.1f} dir={direction:+d}: moved {np.linalg.norm(disp) * 100:5.1f} cm "
              f"(x {disp[0] * 100:+5.1f}, y {disp[1] * 100:+5.1f}), min upright {up:.2f}")
    results.sort(key=lambda r: -r[0])
    best = results[0]
    print(f"\nbest: {best[5]} at {best[3]} Hz, amp {best[4]}, direction {best[6]:+d}: {best[0] * 100:.1f} cm in "
          f"{args.seconds:g} s ({best[0] / args.seconds * 100:.2f} cm/s), min upright {best[2]:.2f}")
    print("Under ~2 cm in 15 s for every pattern means this body/arm/friction combination barely walks even with "
          "a hand-made gait: fix the physics before training.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="octopus.xml")
    ap.add_argument("--mode", choices=["bend", "gait"], default="gait")
    ap.add_argument("--force", type=float, default=None, help="override the muscle peak force (N) for this test")
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()
    (bend_check if args.mode == "bend" else gait_test)(args)


if __name__ == "__main__":
    main()