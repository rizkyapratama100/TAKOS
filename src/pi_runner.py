#!/usr/bin/env python3
"""Run an exported policy on a Raspberry Pi.  Needs only numpy.

    python3 pi_runner.py --policy policy.npz --dry-run obs_log.npz     # no hardware: timing + correctness check
    python3 pi_runner.py --policy policy.npz                           # real loop (after you fill in the two stubs)

Dry run replays the observations recorded on your desktop, runs the network at the real control rate and
compares its actions with the original policy's, so you learn (a) the export is faithful and (b) the Pi keeps up.

IMPORTANT: the policy exported from the current simulation was trained on observations that a real arm cannot
provide (all 23 joint angles, centre-of-mass velocity, muscle activations) and with 23 joint actuators that a
two-tendon arm doesn't have. Hardware mode therefore stays unimplemented on purpose: retrain on the
hardware-matched (tendon-only, encoder + IMU) environment first, then implement read_observation()
and send_to_servos() for your parts.
"""
import argparse
import json
import time

import numpy as np

N_JOINTS = 23
JOINT_LIM = 0.523598776


class Policy:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.cfg = json.loads(str(z["config"]))
        self.mean, self.var = z["obs_mean"], z["obs_var"]
        self.eps, self.clip = float(z["obs_eps"]), float(z["clip_obs"])
        self.hidden = [(z[f"W{i}"], z[f"b{i}"]) for i in range(int(z["n_hidden"]))]
        self.Wo, self.bo = z["Wo"], z["bo"]

    def __call__(self, obs):
        x = np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)
        for W, b in self.hidden:
            x = np.tanh(W @ x + b)
        return np.clip(self.Wo @ x + self.bo, -1.0, 1.0)


class CommandMapper:
    """Same action -> command mapping and low-pass filter as the simulation environment.
    Output vector: [muscle1 (0..1), muscle2 (0..1), 23 joint targets (rad)]."""

    def __init__(self, cfg):
        self.mode, self.smooth = cfg["action_mode"], cfg["smooth"]
        s = np.linspace(0.0, 1.0, N_JOINTS)
        self.basis = np.stack([np.cos(k * np.pi * s) for k in range(cfg["n_modes"])])
        self.ctrl = np.zeros(2 + N_JOINTS)

    def __call__(self, a):
        tgt = np.zeros_like(self.ctrl)
        tgt[:2] = 0.5 * (a[:2] + 1.0)
        joints = a[2:] * JOINT_LIM if self.mode == "full" else np.clip(a[2:] @ self.basis, -1, 1) * JOINT_LIM
        tgt[2:] = joints
        self.ctrl += self.smooth * (tgt - self.ctrl)
        return self.ctrl.copy()


def read_observation():
    raise NotImplementedError(
        "Build the observation EXACTLY as octopus_env.py does (same order, scaling and units) from your "
        "encoders and IMU. This is only possible for a policy trained on a hardware-matched observation.")


def send_to_servos(ctrl):
    raise NotImplementedError(
        "Convert the commands to servo targets (tendon length -> servo angle via your spool radius), "
        "clamp to safe travel, rate-limit, and write them with your servo library.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="policy.npz")
    ap.add_argument("--dry-run", default=None, metavar="OBS_LOG.npz")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    pol = Policy(args.policy)
    mapper = CommandMapper(pol.cfg)
    period = pol.cfg["control_dt"]

    if args.dry_run:
        log = np.load(args.dry_run)
        obs_seq, ref_actions = log["obs"], log["actions"]
        get_obs = lambda i: obs_seq[i % len(obs_seq)]
    else:
        get_obs = lambda i: read_observation()

    compute, errs, overruns = [], [], 0
    t_next = time.perf_counter()
    for i in range(args.steps):
        t0 = time.perf_counter()
        obs = get_obs(i)
        a = pol(obs)
        ctrl = mapper(a)
        if args.dry_run:
            if i < len(ref_actions):
                errs.append(float(np.abs(a - ref_actions[i]).max()))
        else:
            send_to_servos(ctrl)
        compute.append(time.perf_counter() - t0)
        t_next += period
        slack = t_next - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
        else:
            overruns += 1
            t_next = time.perf_counter()

    c = np.array(compute) * 1000
    print(f"{args.steps} steps at a {period * 1000:.0f} ms period: compute {c.mean():.3f} ms mean, "
          f"{c.max():.3f} ms max, {overruns} overruns")
    if errs:
        print(f"max |action - recorded action| = {max(errs):.2e} ({'OK' if max(errs) < 1e-4 else 'MISMATCH'})")


if __name__ == "__main__":
    main()