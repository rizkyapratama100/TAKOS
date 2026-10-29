"""Turn an octopus XML with TWO hinges per link (a1_jx1 + a1_jy1, ...) into one with alternating SINGLE hinges
(jx on odd joints, jy on even joints).  Roughly halves the number of joints, so simulation is faster, and it is
how a printed arm with one pin per joint is built.  Works on hand-edited XML too (comments are kept).

    python alternate_axes.py --xml octopus.xml --out octopus_alt.xml

Each axis then has half the joints, so the arm can curl about half as far in any one direction; check with
    python walk_gait_test.py --xml octopus_alt.xml --mode bend
"""
import argparse
import re
import xml.etree.ElementTree as ET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="octopus.xml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.xml.rsplit(".", 1)[0] + "_alt.xml"

    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(args.xml, parser=parser)
    root = tree.getroot()

    removed = set()
    for body in root.iter("body"):
        for j in list(body.findall("joint")):
            m = re.match(r"(a\d+)_j([xy])(\d+)$", j.get("name", ""))
            if not m:
                continue
            keep = "x" if int(m.group(3)) % 2 == 1 else "y"
            if m.group(2) != keep:
                removed.add(j.get("name"))
                body.remove(j)

    for act in root.findall("actuator"):                   # drop joint actuators that pointed at removed joints
        for a in list(act):
            if a.get("joint") in removed:
                act.remove(a)

    ET.indent(tree)
    tree.write(out)
    # ElementTree drops comments that sit before the root element (e.g. your CAD change notes): copy them back
    head = re.match(r"\s*((?:<!--.*?-->\s*)*)", open(args.xml).read(), re.S).group(1)
    if head.strip():
        body = open(out).read()
        with open(out, "w") as fh:
            fh.write(head.strip() + "\n" + body)
    kept = sum(1 for j in root.iter("joint") if re.match(r"a\d+_j[xy]\d+$", j.get("name", "")))
    print(f"wrote {out}: removed {len(removed)} hinges, {kept} bending hinges remain")


if __name__ == "__main__":
    main()