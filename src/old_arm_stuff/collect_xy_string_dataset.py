import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np


STATIC_ACTIONS = np.array(
    [
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.2, 0.8],
        [0.8, 0.2],
        [0.5, 0.5],
    ],
    dtype=np.float64,
)


def resolve_from_script(path_text: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    path = Path(path_text)
    if not path.is_absolute():
        path = script_dir / path
    return path.resolve()


def tip_xy(data: mujoco.MjData, tip_id: int) -> np.ndarray:
    return np.array([float(data.xpos[tip_id, 0]), float(data.xpos[tip_id, 1])], dtype=np.float64)


def unit_to_model(value: float, unit: str) -> float:
    if unit == "cm":
        return value / 100.0
    return value


def unit_from_model(value: float, unit: str) -> float:
    if unit == "cm":
        return value * 100.0
    return value


def copy_state(model: mujoco.MjModel, src: mujoco.MjData, dst: mujoco.MjData) -> None:
    dst.time = src.time
    dst.qpos[:] = src.qpos
    dst.qvel[:] = src.qvel
    dst.act[:] = src.act
    dst.ctrl[:] = src.ctrl
    dst.qacc_warmstart[:] = src.qacc_warmstart
    if dst.userdata.size:
        dst.userdata[:] = src.userdata
    if dst.mocap_pos.size:
        dst.mocap_pos[:] = src.mocap_pos
    if dst.mocap_quat.size:
        dst.mocap_quat[:] = src.mocap_quat
    mujoco.mj_forward(model, dst)


def estimate_workspace(
    model: mujoco.MjModel,
    tip_id: int,
    rng: np.random.Generator,
    workspace_samples: int,
    hold_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    min_xy = np.array([np.inf, np.inf], dtype=np.float64)
    max_xy = np.array([-np.inf, -np.inf], dtype=np.float64)

    for _ in range(workspace_samples):
        action = rng.random(2)
        for _ in range(hold_steps):
            data.ctrl[0] = float(action[0])
            data.ctrl[1] = float(action[1])
            mujoco.mj_step(model, data)
            xy = tip_xy(data, tip_id)
            min_xy = np.minimum(min_xy, xy)
            max_xy = np.maximum(max_xy, xy)

    return min_xy, max_xy


def parse_targets(target_text: str) -> list[np.ndarray]:
    targets: list[np.ndarray] = []
    chunks = [chunk.strip() for chunk in target_text.split(";") if chunk.strip()]
    if not chunks:
        raise ValueError("Target string is empty. Use format: 'x1,y1;x2,y2'.")
    for chunk in chunks:
        parts = [item.strip() for item in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid target '{chunk}'. Expected 'x,y'.")
        x_val = float(parts[0])
        y_val = float(parts[1])
        targets.append(np.array([x_val, y_val], dtype=np.float64))
    return targets


def build_random_targets(
    rng: np.random.Generator,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    num_targets: int,
    unreachable_fraction: float,
) -> list[tuple[np.ndarray, str]]:
    span = np.maximum(workspace_max - workspace_min, 1e-6)
    reachable_count = int(round(num_targets * (1.0 - unreachable_fraction)))
    reachable_count = min(max(reachable_count, 1), num_targets)
    unreachable_count = num_targets - reachable_count
    targets: list[tuple[np.ndarray, str]] = []

    margin = 0.1
    low = workspace_min + margin * span
    high = workspace_max - margin * span
    high = np.maximum(high, low + 1e-6)
    for _ in range(reachable_count):
        targets.append((rng.uniform(low, high), "reachable"))

    for _ in range(unreachable_count):
        side = int(rng.integers(0, 4))
        offset_x = float(rng.uniform(0.15 * span[0], 0.45 * span[0]))
        offset_y = float(rng.uniform(0.15 * span[1], 0.45 * span[1]))
        if side == 0:
            point = np.array([workspace_max[0] + offset_x, rng.uniform(workspace_min[1], workspace_max[1])])
        elif side == 1:
            point = np.array([workspace_min[0] - offset_x, rng.uniform(workspace_min[1], workspace_max[1])])
        elif side == 2:
            point = np.array([rng.uniform(workspace_min[0], workspace_max[0]), workspace_max[1] + offset_y])
        else:
            point = np.array([rng.uniform(workspace_min[0], workspace_max[0]), workspace_min[1] - offset_y])
        targets.append((point.astype(np.float64), "unreachable"))

    rng.shuffle(targets)
    return targets


def action_pool(prev_action: np.ndarray, rng: np.random.Generator, candidate_actions: int) -> np.ndarray:
    random_actions = rng.random((max(candidate_actions, 1), 2))
    return np.vstack((STATIC_ACTIONS, prev_action.reshape(1, 2), random_actions))


def rollout_tip_for_action(
    model: mujoco.MjModel,
    current_data: mujoco.MjData,
    planner_data: mujoco.MjData,
    tip_id: int,
    action: np.ndarray,
    lookahead_steps: int,
) -> np.ndarray:
    copy_state(model, current_data, planner_data)
    planner_data.ctrl[0] = float(action[0])
    planner_data.ctrl[1] = float(action[1])
    for _ in range(lookahead_steps):
        mujoco.mj_step(model, planner_data)
    return tip_xy(planner_data, tip_id)


def jacobian_guided_action(
    model: mujoco.MjModel,
    current_data: mujoco.MjData,
    planner_data: mujoco.MjData,
    tip_id: int,
    target_xy: np.ndarray,
    prev_action: np.ndarray,
    lookahead_steps: int,
    jacobian_eps: float,
    jacobian_gain: float,
    max_action_step: float,
) -> np.ndarray:
    base_action = np.clip(prev_action, 0.0, 1.0)
    base_tip = rollout_tip_for_action(
        model=model,
        current_data=current_data,
        planner_data=planner_data,
        tip_id=tip_id,
        action=base_action,
        lookahead_steps=lookahead_steps,
    )
    jac = np.zeros((2, 2), dtype=np.float64)

    for i in range(2):
        step_sign = 1.0 if base_action[i] <= (1.0 - jacobian_eps) else -1.0
        pert_action = base_action.copy()
        pert_action[i] = np.clip(base_action[i] + step_sign * jacobian_eps, 0.0, 1.0)
        delta_u = float(pert_action[i] - base_action[i])
        if abs(delta_u) <= 1e-9:
            continue
        pert_tip = rollout_tip_for_action(
            model=model,
            current_data=current_data,
            planner_data=planner_data,
            tip_id=tip_id,
            action=pert_action,
            lookahead_steps=lookahead_steps,
        )
        jac[:, i] = (pert_tip - base_tip) / delta_u

    target_delta = target_xy - base_tip
    du = jacobian_gain * (np.linalg.pinv(jac, rcond=1e-4) @ target_delta)
    du = np.clip(du, -max_action_step, max_action_step)
    return np.clip(base_action + du, 0.0, 1.0)


def choose_action(
    model: mujoco.MjModel,
    current_data: mujoco.MjData,
    planner_data: mujoco.MjData,
    tip_id: int,
    target_xy: np.ndarray,
    current_dist: float,
    prev_action: np.ndarray,
    rng: np.random.Generator,
    candidate_actions: int,
    lookahead_steps: int,
    smooth_weight: float,
    explore_prob: float,
    stagnation_eps: float,
    jacobian_eps: float,
    jacobian_gain: float,
    max_action_step: float,
) -> np.ndarray:
    best_action = prev_action
    best_dist = np.inf
    best_score = np.inf

    guided_action = jacobian_guided_action(
        model=model,
        current_data=current_data,
        planner_data=planner_data,
        tip_id=tip_id,
        target_xy=target_xy,
        prev_action=prev_action,
        lookahead_steps=lookahead_steps,
        jacobian_eps=jacobian_eps,
        jacobian_gain=jacobian_gain,
        max_action_step=max_action_step,
    )
    candidates = np.vstack((action_pool(prev_action, rng, candidate_actions), guided_action.reshape(1, 2)))

    for action in candidates:
        predicted_tip = rollout_tip_for_action(
            model=model,
            current_data=current_data,
            planner_data=planner_data,
            tip_id=tip_id,
            action=action,
            lookahead_steps=lookahead_steps,
        )
        dist = float(np.linalg.norm(predicted_tip - target_xy))
        smooth_penalty = smooth_weight * float(np.linalg.norm(action - prev_action))
        score = dist + smooth_penalty
        if score < best_score:
            best_score = score
            best_dist = dist
            best_action = action.copy()

    if best_dist >= (current_dist - stagnation_eps) and rng.random() < explore_prob:
        return rng.random(2)
    return best_action


def convert_length(length_m: float, length_unit: str) -> float:
    return unit_from_model(length_m, length_unit)


def collect_dataset(
    model: mujoco.MjModel,
    tip_id: int,
    rng: np.random.Generator,
    targets: Iterable[tuple[np.ndarray, str]],
    output_csv: Path,
    metrics_csv: Path,
    sample_steps: int,
    sample_dt: float,
    lookahead_steps: int,
    timeout_s: float,
    target_tolerance: float,
    settle_length_eps: float,
    settle_steps: int,
    candidate_actions: int,
    smooth_weight: float,
    explore_prob: float,
    stagnation_eps: float,
    jacobian_eps: float,
    jacobian_gain: float,
    max_action_step: float,
    target_unit: str,
    length_unit: str,
) -> None:
    data = mujoco.MjData(model)
    planner_data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mujoco.mj_forward(model, planner_data)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file, metrics_csv.open(
        "w", newline="", encoding="utf-8"
    ) as metrics_file:
        rows = csv.writer(csv_file)
        metrics = csv.writer(metrics_file)
        rows.writerow(
            ["target_x", "target_y", "orig_string_l", "orig_string_r", "new_string_l", "new_string_r"]
        )
        metrics.writerow(
            [
                "target_index",
                "target_type",
                "target_x",
                "target_y",
                "status",
                "elapsed_s",
                "steps",
                "best_dist_m",
                "final_dist_m",
                "start_tip_x",
                "start_tip_y",
                "final_tip_x",
                "final_tip_y",
                f"start_string_l_{length_unit}",
                f"start_string_r_{length_unit}",
                f"final_string_l_{length_unit}",
                f"final_string_r_{length_unit}",
            ]
        )

        for idx, (target_xy, target_type) in enumerate(targets, start=1):
            start_tip = tip_xy(data, tip_id)
            start_lengths = data.actuator_length[0:2].copy()
            prev_action = np.clip(data.ctrl[0:2].copy(), 0.0, 1.0)
            elapsed = 0.0
            settled = 0
            reached = False
            best_dist = float(np.linalg.norm(start_tip - target_xy))
            steps = 0
            stabilizing = False
            stabilize_action = prev_action.copy()
            ever_in_tolerance = False

            while elapsed + 1e-12 < timeout_s:
                current_tip = tip_xy(data, tip_id)
                current_dist = float(np.linalg.norm(current_tip - target_xy))
                if stabilizing:
                    action = stabilize_action.copy()
                else:
                    if current_dist <= target_tolerance:
                        action = prev_action.copy()
                    else:
                        action = choose_action(
                            model=model,
                            current_data=data,
                            planner_data=planner_data,
                            tip_id=tip_id,
                            target_xy=target_xy,
                            current_dist=current_dist,
                            prev_action=prev_action,
                            rng=rng,
                            candidate_actions=candidate_actions,
                            lookahead_steps=lookahead_steps,
                            smooth_weight=smooth_weight,
                            explore_prob=explore_prob,
                            stagnation_eps=stagnation_eps,
                            jacobian_eps=jacobian_eps,
                            jacobian_gain=jacobian_gain,
                            max_action_step=max_action_step,
                        )

                old_lengths = data.actuator_length[0:2].copy()
                for _ in range(sample_steps):
                    data.ctrl[0] = float(action[0])
                    data.ctrl[1] = float(action[1])
                    mujoco.mj_step(model, data)

                new_lengths = data.actuator_length[0:2].copy()
                new_tip = tip_xy(data, tip_id)
                dist = float(np.linalg.norm(new_tip - target_xy))
                best_dist = min(best_dist, dist)
                steps += 1
                elapsed += sample_dt

                rows.writerow(
                    [
                        unit_from_model(float(target_xy[0]), target_unit),
                        unit_from_model(float(target_xy[1]), target_unit),
                        convert_length(float(old_lengths[0]), length_unit),
                        convert_length(float(old_lengths[1]), length_unit),
                        convert_length(float(new_lengths[0]), length_unit),
                        convert_length(float(new_lengths[1]), length_unit),
                    ]
                )

                length_delta = float(np.max(np.abs(new_lengths - old_lengths)))
                if dist <= target_tolerance and not stabilizing:
                    ever_in_tolerance = True
                    stabilizing = True
                    stabilize_action = action.copy()
                    settled = 0

                if stabilizing:
                    if length_delta <= settle_length_eps:
                        settled += 1
                    else:
                        settled = 0

                prev_action = action
                if ever_in_tolerance and stabilizing and settled >= settle_steps:
                    reached = True
                    break

            final_dist = float(np.linalg.norm(tip_xy(data, tip_id) - target_xy))
            if reached and final_dist > (target_tolerance * 1.5):
                reached = False

            final_tip = tip_xy(data, tip_id)
            final_lengths = data.actuator_length[0:2].copy()
            status = "reached" if reached else "timeout"
            metrics.writerow(
                [
                    idx,
                    target_type,
                    unit_from_model(float(target_xy[0]), target_unit),
                    unit_from_model(float(target_xy[1]), target_unit),
                    status,
                    elapsed,
                    steps,
                    best_dist,
                    final_dist,
                    unit_from_model(float(start_tip[0]), target_unit),
                    unit_from_model(float(start_tip[1]), target_unit),
                    unit_from_model(float(final_tip[0]), target_unit),
                    unit_from_model(float(final_tip[1]), target_unit),
                    convert_length(float(start_lengths[0]), length_unit),
                    convert_length(float(start_lengths[1]), length_unit),
                    convert_length(float(final_lengths[0]), length_unit),
                    convert_length(float(final_lengths[1]), length_unit),
                ]
            )

            print(
                f"[{idx}] target=({target_xy[0]:.4f}, {target_xy[1]:.4f}) "
                f"type={target_type} status={status} best_dist={best_dist:.4f}m elapsed={elapsed:.2f}s"
            )


def verify_model(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    xml_path: Path,
    tip_id: int,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
) -> dict:
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)
    ]
    issues: list[str] = []
    if tip_id == -1:
        issues.append("Tip body 'link24' was not found.")
    if model.nu < 2:
        issues.append(f"Expected at least 2 actuators, found {model.nu}.")
    if model.nu >= 2:
        expected = {"act1", "act2"}
        missing = sorted(list(expected.difference(actuator_names)))
        if missing:
            issues.append(f"Missing expected actuators: {', '.join(missing)}")
        for i in range(2):
            low = float(model.actuator_ctrlrange[i, 0])
            high = float(model.actuator_ctrlrange[i, 1])
            if low > 0.0 + 1e-9 or high < 1.0 - 1e-9:
                issues.append(
                    f"Actuator {actuator_names[i]} ctrlrange is [{low}, {high}] but expected to include [0, 1]."
                )

    summary = {
        "xml_path": str(xml_path),
        "ok": len(issues) == 0,
        "issues": issues,
        "model": {
            "timestep_s": float(model.opt.timestep),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "actuator_names": actuator_names,
            "initial_tip_xy": tip_xy(data, tip_id).tolist() if tip_id != -1 else None,
            "initial_tendon_lengths_m": data.actuator_length[0:2].astype(float).tolist()
            if model.nu >= 2
            else None,
            "estimated_workspace_xy_min": workspace_min.astype(float).tolist(),
            "estimated_workspace_xy_max": workspace_max.astype(float).tolist(),
        },
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify tentacle_locked.xml assumptions and generate a CSV of string-length transitions while "
            "moving the tip toward target (x, y) coordinates."
        )
    )
    parser.add_argument("--xml", default="tentacle_locked.xml", help="MuJoCo XML path (resolved from script dir).")
    parser.add_argument(
        "--output",
        default="xy_string_dataset.csv",
        help="Transition CSV output path (resolved from script dir).",
    )
    parser.add_argument(
        "--metrics-output",
        default="xy_target_metrics.csv",
        help="Per-target metrics CSV path (resolved from script dir).",
    )
    parser.add_argument(
        "--verification-output",
        default="xy_model_verification.json",
        help="Verification summary JSON path (resolved from script dir).",
    )
    parser.add_argument("--verify-only", action="store_true", help="Only run model verification and exit.")
    parser.add_argument(
        "--targets",
        default="",
        help=(
            "Explicit targets in the format 'x1,y1;x2,y2' in --target-unit units. "
            "If omitted, random targets are generated."
        ),
    )
    parser.add_argument("--num-targets", type=int, default=24, help="Number of random targets.")
    parser.add_argument(
        "--unreachable-fraction",
        type=float,
        default=0.25,
        help="Fraction of random targets sampled outside estimated workspace.",
    )
    parser.add_argument(
        "--sample-dt",
        type=float,
        default=0.1,
        help="Sampling interval in seconds for CSV rows.",
    )
    parser.add_argument(
        "--lookahead-dt",
        type=float,
        default=0.06,
        help="Lookahead duration in seconds used during action selection.",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-target timeout in seconds.")
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=0.03,
        help="Reach tolerance in meters based on Euclidean distance in (x, y).",
    )
    parser.add_argument(
        "--settle-length-eps",
        type=float,
        default=1e-3,
        help="Length-change threshold in meters for considering a sample settled.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=3,
        help="Consecutive settled samples required before declaring target reached.",
    )
    parser.add_argument(
        "--candidate-actions",
        type=int,
        default=96,
        help="Number of random candidate tendon commands tested per sample.",
    )
    parser.add_argument(
        "--smooth-weight",
        type=float,
        default=0.03,
        help="Penalty weight for abrupt tendon command changes.",
    )
    parser.add_argument(
        "--explore-prob",
        type=float,
        default=0.08,
        help="Chance of random exploration when progress stalls.",
    )
    parser.add_argument(
        "--stagnation-eps",
        type=float,
        default=1e-4,
        help="Minimum distance improvement considered meaningful.",
    )
    parser.add_argument(
        "--jacobian-eps",
        type=float,
        default=0.04,
        help="Finite-difference delta for Jacobian-guided action proposal.",
    )
    parser.add_argument(
        "--jacobian-gain",
        type=float,
        default=0.8,
        help="Step gain for Jacobian-guided action update.",
    )
    parser.add_argument(
        "--max-action-step",
        type=float,
        default=0.25,
        help="Maximum per-sample change per tendon command.",
    )
    parser.add_argument(
        "--workspace-samples",
        type=int,
        default=450,
        help="Number of random controls used to estimate workspace.",
    )
    parser.add_argument(
        "--workspace-hold-steps",
        type=int,
        default=80,
        help="MuJoCo steps to hold each random control during workspace estimation.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--length-unit",
        choices=["m", "cm"],
        default="cm",
        help="Unit used in output CSVs for string lengths.",
    )
    parser.add_argument(
        "--target-unit",
        choices=["m", "cm"],
        default="cm",
        help="Unit used for target coordinates in inputs and output CSVs.",
    )
    args = parser.parse_args()

    if not (0.0 <= args.unreachable_fraction <= 1.0):
        raise ValueError("--unreachable-fraction must be between 0 and 1.")
    if args.num_targets < 1:
        raise ValueError("--num-targets must be at least 1.")

    xml_path = resolve_from_script(args.xml)
    output_csv = resolve_from_script(args.output)
    metrics_csv = resolve_from_script(args.metrics_output)
    verification_json = resolve_from_script(args.verification_output)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link24")

    rng = np.random.default_rng(args.seed)
    workspace_min, workspace_max = estimate_workspace(
        model=model,
        tip_id=tip_id,
        rng=rng,
        workspace_samples=args.workspace_samples,
        hold_steps=args.workspace_hold_steps,
    )
    summary = verify_model(
        model=model,
        data=data,
        xml_path=xml_path,
        tip_id=tip_id,
        workspace_min=workspace_min,
        workspace_max=workspace_max,
    )
    workspace_min_display = np.array(
        [unit_from_model(float(workspace_min[0]), args.target_unit), unit_from_model(float(workspace_min[1]), args.target_unit)]
    )
    workspace_max_display = np.array(
        [unit_from_model(float(workspace_max[0]), args.target_unit), unit_from_model(float(workspace_max[1]), args.target_unit)]
    )
    print(
        "Estimated workspace x/y in "
        f"{args.target_unit}: min=({workspace_min_display[0]:.3f}, {workspace_min_display[1]:.3f}) "
        f"max=({workspace_max_display[0]:.3f}, {workspace_max_display[1]:.3f})"
    )

    verification_json.parent.mkdir(parents=True, exist_ok=True)
    verification_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Verification summary written to: {verification_json}")
    if summary["ok"]:
        print("Model verification: OK")
    else:
        print("Model verification: FAILED")
        for issue in summary["issues"]:
            print(f"  - {issue}")
        return 1

    if args.verify_only:
        return 0

    timestep = float(model.opt.timestep)
    sample_steps = max(1, int(round(args.sample_dt / timestep)))
    lookahead_steps = max(1, int(round(args.lookahead_dt / timestep)))
    actual_sample_dt = sample_steps * timestep
    print(
        f"Sampling at {actual_sample_dt:.6f}s ({sample_steps} MuJoCo steps per row), "
        f"lookahead {lookahead_steps} steps."
    )

    if args.targets:
        targets = [
            (
                np.array(
                    [unit_to_model(float(point[0]), args.target_unit), unit_to_model(float(point[1]), args.target_unit)],
                    dtype=np.float64,
                ),
                "manual",
            )
            for point in parse_targets(args.targets)
        ]
    else:
        targets = build_random_targets(
            rng=rng,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            num_targets=args.num_targets,
            unreachable_fraction=args.unreachable_fraction,
        )

    collect_dataset(
        model=model,
        tip_id=tip_id,
        rng=rng,
        targets=targets,
        output_csv=output_csv,
        metrics_csv=metrics_csv,
        sample_steps=sample_steps,
        sample_dt=actual_sample_dt,
        lookahead_steps=lookahead_steps,
        timeout_s=float(args.timeout),
        target_tolerance=float(args.target_tolerance),
        settle_length_eps=float(args.settle_length_eps),
        settle_steps=int(args.settle_steps),
        candidate_actions=int(args.candidate_actions),
        smooth_weight=float(args.smooth_weight),
        explore_prob=float(args.explore_prob),
        stagnation_eps=float(args.stagnation_eps),
        jacobian_eps=float(args.jacobian_eps),
        jacobian_gain=float(args.jacobian_gain),
        max_action_step=float(args.max_action_step),
        target_unit=args.target_unit,
        length_unit=args.length_unit,
    )

    print(f"Transition CSV written to: {output_csv}")
    print(f"Metrics CSV written to: {metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
