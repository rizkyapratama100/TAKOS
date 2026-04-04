import argparse
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

from generate_tentacle_xml import transform_xml


def load_generate_mujoco_xml(design_tool_dir: Path):
    xml_generator_path = design_tool_dir / "xml_generator.py"
    if not xml_generator_path.exists():
        raise FileNotFoundError(f"Missing upstream file: {xml_generator_path}")

    spec = importlib.util.spec_from_file_location(
        "openspirobs_xml_generator",
        xml_generator_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from: {xml_generator_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generate_fn = getattr(module, "generate_mujoco_xml", None)
    if generate_fn is None:
        raise AttributeError(
            f"'generate_mujoco_xml' not found in {xml_generator_path}"
        )
    return generate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Headless pipeline: run Open Spiral Robots xml_generator.py, then apply "
            "TAKOS profile tweaks (locked/crawl + stiffness/damping scaling)."
        )
    )
    parser.add_argument(
        "--openspirobs-design-tool",
        required=True,
        type=Path,
        help="Path to Open-Spiral-Robots/design-tool directory.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output XML path.")
    parser.add_argument(
        "--profile",
        choices=["locked", "crawl"],
        default="locked",
        help="TAKOS output profile.",
    )

    # Upstream xml_generator.py inputs
    parser.add_argument(
        "--stl-name",
        default="baselink.stl",
        help="STL mesh filename referenced by generated XML.",
    )
    parser.add_argument(
        "--unit-height",
        type=float,
        required=True,
        help="Unit height in mm for upstream generator.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        required=True,
        help=(
            "Upstream scale parameter (their gamma). "
            "Per-link scale step = 1/scale."
        ),
    )
    parser.add_argument(
        "--num-units",
        type=int,
        default=24,
        help="Number of links/units to generate.",
    )
    parser.add_argument(
        "--joint-type",
        choices=["hinge", "ball"],
        default="hinge",
        help="Joint type for upstream generator.",
    )
    parser.add_argument(
        "--joint-limit-deg",
        type=float,
        default=None,
        help="Optional hinge limit in degrees for upstream generator.",
    )
    parser.add_argument(
        "--robot-length",
        type=float,
        required=True,
        help="Robot length in mm for upstream generator.",
    )
    parser.add_argument(
        "--site-points",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        required=True,
        help="Site points in mm passed to upstream generator.",
    )
    parser.add_argument(
        "--cable-mode",
        choices=[2, 3],
        type=int,
        default=2,
        help="Upstream cable mode. TAKOS runtime currently expects 2-cable outputs.",
    )

    # TAKOS transform inputs
    parser.add_argument(
        "--stiffness-scale",
        type=float,
        default=1.0,
        help="Multiply all joint stiffness values by this factor.",
    )
    parser.add_argument(
        "--damping-scale",
        type=float,
        default=1.0,
        help="Multiply all joint damping values by this factor.",
    )
    parser.add_argument(
        "--stiffness-exponent",
        type=float,
        default=1.0,
        help="Exponential multiplier (pow(exp, i)) for joint stiffness along the chain (0=base).",
    )
    parser.add_argument(
        "--damping-exponent",
        type=float,
        default=1.0,
        help="Exponential multiplier (pow(exp, i)) for joint damping along the chain (0=base).",
    )
    parser.add_argument(
        "--actuator-type",
        choices=["muscle", "position"],
        default="muscle",
        help="Type of tendon actuator.",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=1000.0,
        help="Proportional gain for position actuators.",
    )
    parser.add_argument(
        "--length-range-scale",
        type=float,
        default=1.0,
        help="Scale the allowed contraction range of the tendons.",
    )
    parser.add_argument(
        "--muscle-force",
        type=float,
        default=None,
        help="Manual override for muscle force.",
    )
    parser.add_argument(
        "--joint-position-kp",
        type=float,
        default=10.0,
        help="Position actuator kp when using crawl profile.",
    )
    parser.add_argument(
        "--save-raw-xml",
        type=Path,
        default=None,
        help="Optional path to save pre-transform upstream XML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_mujoco_xml = load_generate_mujoco_xml(args.openspirobs_design_tool)

    site_points = tuple(args.site_points)

    with TemporaryDirectory(prefix="takos_xml_pipeline_") as tmp_dir:
        raw_xml = Path(tmp_dir) / "raw_upstream.xml"

        generate_mujoco_xml(
            output_path=raw_xml,
            stl_name=args.stl_name,
            unit_height=args.unit_height,
            scale=args.scale,
            num_units=args.num_units,
            joint_type=args.joint_type,
            joint_limit_deg=args.joint_limit_deg,
            robot_length=args.robot_length,
            site_points=site_points,
            cable_mode=args.cable_mode,
        )

        if args.save_raw_xml is not None:
            args.save_raw_xml.parent.mkdir(parents=True, exist_ok=True)
            args.save_raw_xml.write_text(raw_xml.read_text(encoding="utf-8"), encoding="utf-8")

        transform_xml(
            input_xml=raw_xml,
            output_xml=args.output,
            profile=args.profile,
            stiffness_scale=args.stiffness_scale,
            damping_scale=args.damping_scale,
            stiffness_exponent=args.stiffness_exponent,
            damping_exponent=args.damping_exponent,
            actuator_type=args.actuator_type,
            kp=args.kp,
            length_range_scale=args.length_range_scale,
            muscle_force=args.muscle_force,
            joint_position_kp=args.joint_position_kp,
        )


if __name__ == "__main__":
    main()
