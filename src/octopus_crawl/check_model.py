"""Sanity-check the MJCF before training (masses, resting pose, can the muscles actually bend the arm?).

    python check_model.py --xml spiral_robot.xml
"""
import argparse
import time

import mujoco
import numpy as np


def run(m, d, ctrl, seconds):
    mujoco.mj_resetData(m, d)
    d.ctrl[:] = ctrl
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="spiral_robot.xml")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"j{i}") for i in range(1, 24)]
    qadr = m.jnt_qposadr[jids]
    mu = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ("act1", "act2")]

    total = m.body_mass.sum()
    print(f"nu={m.nu} na={m.na} nbody={m.nbody} ntendon={m.ntendon}")
    print(f"total mass {total:.3f} kg (weight {total * 9.81:.1f} N); "
          f"base link {m.body_mass[1]:.3f} kg, tip link {m.body_mass[-1]:.4f} kg")
    print(f"muscle peak force {m.actuator_gainprm[mu, 2]} N")

    run(m, d, np.zeros(m.nu), 1.0)
    print(f"\n[rest, 1 s] base z={d.qpos[2]:.3f} m  contacts={d.ncon}  max|joint|={np.abs(d.qpos[qadr]).max():.3f} rad")
    print(f"            tendon lengths {d.ten_length}  allowed range {m.tendon_range.tolist()}")

    # The joint position actuators (kp=10) hold every joint at its target and would fight the
    # muscles, so switch them off for this muscle-only test and restore them afterwards.
    adj = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"adj{i}") for i in range(1, 24)]
    saved_gain, saved_bias = m.actuator_gainprm[adj].copy(), m.actuator_biasprm[adj].copy()
    m.actuator_gainprm[adj] = 0.0
    m.actuator_biasprm[adj] = 0.0
    for k, name in enumerate(("tendon1 (top)", "tendon2 (bottom)")):
        ctrl = np.zeros(m.nu)
        ctrl[mu[k]] = 1.0
        run(m, d, ctrl, 1.5)
        q = d.qpos[qadr]
        total = q.sum()
        print(f"[{name} full activation, muscle only, 1.5 s] total bend {total:+.3f} rad "
              f"({np.degrees(total):+.0f} deg), max|joint| {np.abs(q).max():.3f}, "
              f"tip height {d.xpos[-1][2]:.3f} m")
        print("   per-joint angles (rad):", np.round(q, 2).tolist())
        if abs(total) < 1.0:
            print("   NOTE: total bend under ~1 rad (57 deg). Crawling needs bigger curls: raise muscle "
                  "`force`, lower density, or lower joint stiffness (see patch_model.py).")
    m.actuator_gainprm[adj] = saved_gain
    m.actuator_biasprm[adj] = saved_bias

    mujoco.mj_resetData(m, d)
    t0, n = time.time(), 2000
    for _ in range(n):
        d.ctrl[:] = np.random.uniform(-0.2, 0.5, m.nu)
        mujoco.mj_step(m, d)
    print(f"\nspeed: {n / (time.time() - t0):.0f} physics steps/s (control step = 20 of these)")


if __name__ == "__main__":
    main()
