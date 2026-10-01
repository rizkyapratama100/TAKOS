"""Why doesn't the arm move?  Reports self-collisions, a per-joint torque budget
(weight of the distal chain vs. what the top muscle can deliver) and motion tests.

    python diagnose.py --xml spiral_robot.xml
"""
import argparse

import mujoco
import numpy as np


def bname(m, i):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"#{i}"


def settle(m, d, ctrl=None, seconds=1.0):
    mujoco.mj_resetData(m, d)
    d.ctrl[:] = 0.0 if ctrl is None else ctrl
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="spiral_robot.xml")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"j{i}") for i in range(1, 24)]
    qadr, dadr = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
    mu = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ("act1", "act2")]
    adj = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"adj{i}") for i in range(1, 24)]
    ground = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")

    # ---- 1. contacts at rest
    settle(m, d)
    n_ground, pairs = 0, {}
    for c in d.contact[: d.ncon]:
        if ground in (c.geom1, c.geom2):
            n_ground += 1
        else:
            key = tuple(sorted((bname(m, m.geom_bodyid[c.geom1]), bname(m, m.geom_bodyid[c.geom2]))))
            pairs[key] = min(pairs.get(key, 0.0), c.dist)
    print(f"[1] contacts at rest: {d.ncon} total, {n_ground} with ground, "
          f"{d.ncon - n_ground} between links ({len(pairs)} link pairs)")
    for (a, b), dist in list(pairs.items())[:10]:
        print(f"      {a} <-> {b}   penetration {-dist * 1000:.2f} mm")
    if pairs:
        print("    => links collide with each other: this likely locks the joints. "
              "patch_model.py disables link-link collision.")

    # ---- 2. torque budget
    mujoco.mj_forward(m, d)
    base = d.qfrc_actuator[dadr].copy()
    muscle_torque = []
    for k in range(2):
        d.act[:] = 0.0
        d.act[k] = 1.0
        d.ctrl[:] = 0.0
        d.ctrl[mu[k]] = 1.0
        mujoco.mj_forward(m, d)
        muscle_torque.append(np.abs(d.qfrc_actuator[dadr] - base))
    d.act[:] = 0.0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)

    print("\n[2] torque budget per joint (N*m).  ratio = muscle / (weight + spring at 15 deg); >1 can lift")
    print("    joint  distal_mass_g  weight  spring  muscle1  muscle2  ratio")
    for k, jid in enumerate(jids):
        child = m.jnt_bodyid[jid]
        mass = m.body_subtreemass[child]
        lever = abs(d.subtree_com[child][0] - d.xanchor[jid][0])
        weight = mass * 9.81 * lever
        spring = m.jnt_stiffness[jid] * np.radians(15)
        ratio = muscle_torque[0][k] / (weight + spring + 1e-9)
        print(f"    j{k + 1:<4d} {mass * 1000:12.1f} {weight:7.3f} {spring:7.3f} "
              f"{muscle_torque[0][k]:8.3f} {muscle_torque[1][k]:8.3f} {ratio:6.2f}{'  OK' if ratio > 1 else ''}")

    # ---- 3. motion tests
    print("\n[3] motion tests (1.5 s)")
    for k, name in enumerate(("muscle 1 (top)", "muscle 2 (bottom)")):
        ctrl = np.zeros(m.nu)
        ctrl[mu[k]] = 1.0
        settle(m, d, ctrl, 1.5)
        q = d.qpos[qadr]
        print(f"    {name:18s} full: mean angle {q.mean():+.3f} rad, max|q| {np.abs(q).max():.3f}")
    ctrl = np.zeros(m.nu)
    ctrl[adj] = 0.3
    settle(m, d, ctrl, 1.5)
    q = d.qpos[qadr]
    print(f"    joint actuators +0.3 rad: mean angle {q.mean():+.3f} rad, max|q| {np.abs(q).max():.3f} "
          f"(commanded 0.300)")
    print("    If the joint actuators also fail to bend the arm, the cause is collisions/contacts, not muscle strength.")


if __name__ == "__main__":
    main()
