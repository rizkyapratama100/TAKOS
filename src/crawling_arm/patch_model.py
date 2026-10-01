"""Write a modified copy of the MJCF so the arm can actually move.

    python patch_model.py --xml spiral_robot.xml            # -> spiral_robot_v2.xml
    python patch_model.py --xml spiral_robot.xml --density 1100 --force 30

What it changes (the original file is never touched):
  * geom density            (5000 is steel-like; the default here is 1100, plastic/silicone-like)
  * muscle peak force       (10 N -> --force)
  * joint stiffness / damping / frictionloss, scaled by the given factors
  * link-link collision off (links still collide with the ground)

These are starting values to get the model moving in simulation. Replace density and force
with the measured mass and tendon force of the real arm once you have them.
"""
import argparse
import os
import xml.etree.ElementTree as ET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="spiral_robot.xml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--density", type=float, default=1100.0)
    ap.add_argument("--force", type=float, default=30.0)
    ap.add_argument("--stiffness-scale", type=float, default=0.3)
    ap.add_argument("--damping-scale", type=float, default=0.5)
    ap.add_argument("--frictionloss-scale", type=float, default=0.3)
    ap.add_argument("--keep-self-collision", action="store_true")
    args = ap.parse_args()

    out = args.out or os.path.splitext(args.xml)[0] + "_v2.xml"
    tree = ET.parse(args.xml)
    root = tree.getroot()

    default_geom = root.find("default/geom")
    default_geom.set("density", f"{args.density:g}")
    if not args.keep_self_collision:
        # links: contype 0 / conaffinity 1 -> never collide with each other, still hit the ground
        default_geom.set("contype", "0")
        default_geom.set("conaffinity", "1")
        ground = root.find("worldbody/geom[@name='ground']")
        ground.set("contype", "1")
        ground.set("conaffinity", "1")

    scales = {"stiffness": args.stiffness_scale, "damping": args.damping_scale,
              "frictionloss": args.frictionloss_scale}
    n_joints = 0
    for j in root.iter("joint"):
        if not j.get("name", "").startswith("j"):
            continue
        n_joints += 1
        for attr, s in scales.items():
            if j.get(attr) is not None:
                j.set(attr, f"{float(j.get(attr)) * s:.6f}")

    n_muscles = 0
    for mu in root.iter("muscle"):
        mu.set("force", f"{args.force:g}")
        n_muscles += 1

    ET.indent(tree)
    tree.write(out)
    print(f"wrote {out}: density={args.density:g}, {n_muscles} muscles at {args.force:g} N, "
          f"{n_joints} joints scaled {scales}, self-collision {'kept' if args.keep_self_collision else 'off'}")


if __name__ == "__main__":
    main()
