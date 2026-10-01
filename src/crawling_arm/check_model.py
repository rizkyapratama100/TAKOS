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

    for k, name in enumerate(("tendon1 (top)", "tendon2 (bottom)")):
        ctrl = np.zeros(m.nu)
        ctrl[mu[k]] = 1.0
        run(m, d, ctrl, 1.5)
        q = d.qpos[qadr]
        print(f"[{name} full activation, 1.5 s] mean joint angle {q.mean():+.3f} rad, "
              f"max|joint| {np.abs(q).max():.3f}, tip height {d.xpos[-1][2]:.3f} m")
        if np.abs(q).mean() < 0.03:
            print("   WARNING: barely moves. The arm is probably too heavy/stiff for 10 N muscles; "
                  "raise muscle `force`, lower geom density, or lower joint stiffness.")

    mujoco.mj_resetData(m, d)
    t0, n = time.time(), 2000
    for _ in range(n):
        d.ctrl[:] = np.random.uniform(-0.2, 0.5, m.nu)
        mujoco.mj_step(m, d)
    print(f"\nspeed: {n / (time.time() - t0):.0f} physics steps/s (control step = 20 of these)")


if __name__ == "__main__":
    main()
