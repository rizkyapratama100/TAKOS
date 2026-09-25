import argparse
import copy
from pathlib import Path
import re
import xml.etree.ElementTree as ET


PROFILE_SETTINGS = {
    "locked": {
        "gravity": "0 0 0",
        "base_pos": "0 0 0.5",
        "base_quat": "0.70710678 0 0.70710678 0",
        "freejoint": False,
        "default_muscle_force": 30.0,
        "include_position_actuators": False,
    },
    "crawl": {
        "gravity": "0 0 -9.81",
        "base_pos": "0 0 0.05",
        "base_quat": "0.5 0.5 0.5 0.5",
        "freejoint": True,
        "default_muscle_force": 10.0,
        "include_position_actuators": True,
    },
}


def format_float(value: float, digits: int = 6) -> str:
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def find_required(root: ET.Element, path: str, description: str) -> ET.Element:
    element = root.find(path)
    if element is None:
        raise ValueError(f"Missing required XML node: {description} (path: {path})")
    return element


def configure_option_and_base(root: ET.Element, profile: str) -> None:
    settings = PROFILE_SETTINGS[profile]

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("gravity", "0 0 0")
    option.set("timestep", "0.001")
    option.set("solver", "Newton")

    geom = root.find("./default/geom")
    if geom is not None:
        geom.set("density", "5000")
        geom.set("friction", "1 0.5 0.5")
        # Robust collision settings to prevent "clipping" through itself
        geom.set("contype", "1")
        geom.set("conaffinity", "1")
        geom.set("solimp", "0.99 0.99 0.01")
        geom.set("solref", "0.01 1")

    link1 = find_required(root, "./worldbody/body[@name='link1']", "worldbody/link1")
    link1.set("pos", settings["base_pos"])
    link1.set("quat", settings["base_quat"])

    freejoints = [child for child in list(link1) if child.tag == "freejoint"]
    if settings["freejoint"]:
        if not freejoints:
            link1.insert(0, ET.Element("freejoint"))
        elif len(freejoints) > 1:
            for extra in freejoints[1:]:
                link1.remove(extra)
    else:
        for freejoint in freejoints:
            link1.remove(freejoint)


def get_chain_joint_elements(root: ET.Element) -> list[ET.Element]:
    link1 = find_required(root, "./worldbody/body[@name='link1']", "worldbody/link1")
    body_name_pattern = re.compile(r"link(\d+)$")
    indexed_joints: list[tuple[int, ET.Element]] = []

    for body in link1.iter("body"):
        body_name = body.get("name", "")
        match = body_name_pattern.fullmatch(body_name)
        if match is None:
            continue

        link_index = int(match.group(1))
        if link_index <= 1:
            continue

        joint = next((child for child in list(body) if child.tag == "joint"), None)
        if joint is None:
            continue

        indexed_joints.append((link_index - 1, joint))

    indexed_joints.sort(key=lambda pair: pair[0])
    return [joint for _, joint in indexed_joints]


def canonicalize_chain_joints(root: ET.Element) -> list[str]:
    joint_names: list[str] = []
    for i, joint in enumerate(get_chain_joint_elements(root), start=1):
        joint_name = f"j{i}"
        joint_names.append(joint_name)
        joint.set("name", joint_name)

        joint_type = joint.get("type")
        if joint_type is None:
            joint_type = "hinge"
            joint.set("type", joint_type)

        if joint_type == "hinge":
            if "axis" not in joint.attrib:
                joint.set("axis", "1 0 0")
            if "range" not in joint.attrib:
                joint.set("range", "-0.523598776 0.523598776")

    return joint_names


def apply_joint_tuning(
    root: ET.Element,
    stiffness_scale: float,
    damping_scale: float,
    stiffness_exponent: float = 1.0,
    damping_exponent: float = 1.0,
) -> None:
    for i, joint in enumerate(get_chain_joint_elements(root)):
        if "stiffness" in joint.attrib:
            stiffness = float(joint.get("stiffness", "0"))
            new_val = stiffness * stiffness_scale * (stiffness_exponent ** i)
            joint.set("stiffness", format_float(new_val))
        if "damping" in joint.attrib:
            damping = float(joint.get("damping", "0"))
            new_val = damping * damping_scale * (damping_exponent ** i)
            joint.set("damping", format_float(new_val))


def ensure_joint_aux_params(root: ET.Element) -> None:
    for joint in get_chain_joint_elements(root):
        if "frictionloss" not in joint.attrib:
            if "stiffness" in joint.attrib:
                value = 0.1 * float(joint.get("stiffness", "0"))
            else:
                value = 0.05
            joint.set("frictionloss", format_float(value))

        if "armature" not in joint.attrib:
            if "damping" in joint.attrib:
                value = 0.01 * float(joint.get("damping", "0"))
            else:
                value = 0.01
            joint.set("armature", format_float(value))


def configure_actuators(
    root: ET.Element,
    muscle_force: float | None,
    joint_position_kp: float,
    actuator_type: str = "muscle",
    kp: float = 1000.0,
    length_range_scale: float = 1.0,
) -> None:
    actuators = root.find("actuator")
    if actuators is None:
        actuators = ET.SubElement(root, "actuator")
    else:
        actuators.clear()

    tendons = root.findall("./tendon/spatial")
    for i, tendon in enumerate(tendons):
        name = tendon.get("name")
        length_range_str = tendon.get("range")
        if length_range_str:
            l_min, l_max = map(float, length_range_str.split())
            center = (l_min + l_max) / 2
            half_width = (l_max - l_min) / 2 * length_range_scale
            l_min, l_max = center - half_width, center + half_width
            tendon.set("range", f"{format_float(l_min)} {format_float(l_max)}")
        else:
            l_min, l_max = 0.4, 1.0

        if actuator_type == "muscle":
            act = ET.SubElement(actuators, "muscle")
            act.set("name", f"act{i+1}")
            act.set("tendon", name)
            act.set("ctrllimited", "true")
            act.set("ctrlrange", "0 1")
            if muscle_force is not None:
                act.set("force", format_float(muscle_force))
            act.set("lengthrange", f"{format_float(l_min)} {format_float(l_max)}")
        elif actuator_type == "position":
            act = ET.SubElement(actuators, "general")
            act.set("name", f"act{i+1}")
            act.set("tendon", name)
            act.set("ctrllimited", "true")
            act.set("ctrlrange", "0 1")
            act.set("gaintype", "fixed")
            gain = -kp * (l_max - l_min)
            act.set("gainprm", format_float(gain))
            act.set("biastype", "affine")
            act.set("biasprm", f"{format_float(kp * l_max)} {format_float(-kp)} 0")


def build_position_actuators(
    joint_names: list[str],
    joint_position_kp: float,
) -> list[ET.Element]:
    actuators = []
    for i, joint_name in enumerate(joint_names, start=1):
        actuator = ET.Element("position")
        actuator.set("name", f"adj{i}")
        actuator.set("joint", joint_name)
        actuator.set("kp", format_float(joint_position_kp))
        actuator.set("ctrlrange", "-0.523599 0.523599")
        actuator.set("forcerange", "-1 1")
        actuators.append(actuator)
    return actuators


def transform_xml(
    input_xml: Path,
    output_xml: Path,
    profile: str,
    stiffness_scale: float,
    damping_scale: float,
    stiffness_exponent: float,
    damping_exponent: float,
    actuator_type: str,
    kp: float,
    length_range_scale: float,
    muscle_force: float | None,
    joint_position_kp: float,
) -> None:
    tree = ET.parse(input_xml)
    root = tree.getroot()

    if root.tag != "mujoco":
        raise ValueError(f"Unexpected root tag '{root.tag}'. Expected 'mujoco'.")

    configure_option_and_base(root, profile)
    joint_names = canonicalize_chain_joints(root)
    apply_joint_tuning(
        root,
        stiffness_scale,
        damping_scale,
        stiffness_exponent,
        damping_exponent,
    )
    ensure_joint_aux_params(root)
    configure_actuators(
        root,
        muscle_force,
        joint_position_kp,
        actuator_type,
        kp,
        length_range_scale,
    )

    if PROFILE_SETTINGS[profile]["include_position_actuators"]:
        actuator_node = root.find("actuator")
        for position_actuator in build_position_actuators(joint_names, joint_position_kp):
            actuator_node.append(position_actuator)

    ET.indent(tree, space="  ")
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transform Open Spiral Robots-style MuJoCo XML into TAKOS profiles "
            "with programmatic stiffness/damping tuning."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to source MuJoCo XML (e.g., Open Spiral Robots export).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to output XML file.",
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=sorted(PROFILE_SETTINGS.keys()),
        help="TAKOS XML profile to apply.",
    )
    parser.add_argument(
        "--stiffness-scale",
        type=float,
        default=1.0,
        help="Multiply every joint stiffness by this factor.",
    )
    parser.add_argument(
        "--damping-scale",
        type=float,
        default=1.0,
        help="Multiply every joint damping by this factor.",
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
        help="Override tendon muscle force. Defaults to profile value.",
    )
    parser.add_argument(
        "--joint-position-kp",
        type=float,
        default=10.0,
        help="Position actuator kp for crawl profile.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transform_xml(
        input_xml=args.input,
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
