"""Sanity-check an STL exported from CAD (Fusion 360 etc.) before using it in MuJoCo.  numpy only.

    python check_stl.py baselink.stl
    python check_stl.py my_segment_link.stl --scale 0.001 --density 1100

Reports: units guess, size, where the origin sits relative to the part, watertightness, orientation
(inverted/mirrored meshes have negative volume and make MuJoCo fail to compile), and the mass you'd get.

What MuJoCo expects of a link mesh in THIS project:
  * origin at the joint pivot (base of the link), arm axis along +z, part centred on x=0, y=0;
  * units: the XML uses scale 0.001, i.e. the STL is in millimetres;
  * one watertight solid body, outward-facing normals, a few thousand triangles at most.
"""
import argparse
import re
import struct

import numpy as np


def load_stl(path):
    data = open(path, "rb").read()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + 50 * n:
            dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            return np.frombuffer(data, dtype=dt, count=n, offset=84)["v"].astype(np.float64)
    text = data.decode("ascii", errors="ignore")
    vs = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
    if not vs:
        raise SystemExit("could not parse this STL (neither binary nor ASCII)")
    return np.array(vs, dtype=float).reshape(-1, 3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stl")
    ap.add_argument("--scale", type=float, default=None, help="STL units -> metres (default: guessed)")
    ap.add_argument("--density", type=float, default=1100.0)
    args = ap.parse_args()

    tris = load_stl(args.stl)
    verts = tris.reshape(-1, 3)
    lo, hi = verts.min(0), verts.max(0)
    size = hi - lo
    diag = float(np.linalg.norm(size))
    print(f"{args.stl}: {len(tris)} triangles")
    print(f"bounding box (file units): x [{lo[0]:.3f}, {hi[0]:.3f}]  y [{lo[1]:.3f}, {hi[1]:.3f}]  z [{lo[2]:.3f}, {hi[2]:.3f}]")
    print(f"size (file units): {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f}")

    big = float(size.max())
    if args.scale is None:
        args.scale = 0.001 if big > 5 else (0.01 if big > 0.5 else 1.0)
        unit = {0.001: "millimetres", 0.01: "centimetres", 1.0: "metres"}[args.scale]
        print(f"units: largest side {big:.3g} -> looks like {unit}; use scale=\"{args.scale:g} {args.scale:g} {args.scale:g}\" "
              f"in the <mesh> tag ({size.max() * args.scale * 1000:.1f} mm long after scaling)")
    else:
        print(f"units: using your --scale {args.scale:g} ({size.max() * args.scale * 1000:.1f} mm longest side)")

    centre_xy = (lo[:2] + hi[:2]) / 2
    print(f"origin: ({0:.0f}, {0:.0f}, {0:.0f}); part centre x,y = ({centre_xy[0]:.3f}, {centre_xy[1]:.3f}), z range [{lo[2]:.3f}, {hi[2]:.3f}]")
    if np.abs(centre_xy).max() > 0.05 * max(size[0], size[1]):
        print("   PROBLEM? part is not centred on x=0,y=0: the arm axis should pass through the origin along +z "
              "(move/rotate the part in CAD, or the whole chain will be offset and twisted).")
    if lo[2] < -0.05 * size[2] or lo[2] > 0.05 * size[2]:
        print("   NOTE: the base of the link is not at z=0. Joints sit at the body origin, so put the pivot at the origin in CAD.")

    # weld vertices, then check the surface
    keys = np.round(verts / (diag * 1e-6)).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)
    area2 = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    degenerate = int((area2 < 1e-12 * diag * diag).sum())

    und = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(und, axis=0, return_counts=True)
    boundary, nonmanifold = int((counts == 1).sum()), int((counts > 2).sum())
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    nv = int(inv.max()) + 1
    _, dcounts = np.unique(directed[:, 0].astype(np.int64) * nv + directed[:, 1], return_counts=True)
    inconsistent = int((dcounts > 1).sum())

    print(f"\nwatertight: {'OK' if boundary == 0 else f'PROBLEM, {boundary} open edges (holes/gaps)'}")
    print(f"manifold:   {'OK' if nonmanifold == 0 else f'PROBLEM, {nonmanifold} edges shared by 3+ faces (touching or overlapping bodies)'}")
    print(f"orientation consistent: {'OK' if inconsistent == 0 else f'PROBLEM, {inconsistent} edges with flipped faces'}")
    print(f"degenerate triangles: {degenerate}")

    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    vol6 = np.einsum("ij,ij->i", v0, np.cross(v1, v2))
    volume = vol6.sum() / 6.0
    print(f"signed volume: {volume:.4g} (file units^3)  -> "
          f"{'OK' if volume > 0 else 'PROBLEM: negative = inverted or mirrored mesh; MuJoCo will refuse it. Re-export, or flip normals'}")
    vol_m3 = abs(volume) * args.scale ** 3
    print(f"solid volume {vol_m3 * 1e6:.2f} cm^3 -> {vol_m3 * args.density * 1000:.1f} g at density {args.density:g} kg/m^3")
    com = ((v0 + v1 + v2) / 4.0 * vol6[:, None]).sum(0) / vol6.sum() if vol6.sum() else np.zeros(3)
    print(f"centre of mass (file units): ({com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f})")
    if len(tris) > 20000:
        print("NOTE: very dense mesh; simplify/export at a lower refinement (MuJoCo only uses the convex hull for collision).")
    print("\nTip: if the CAD part has hollow channels or cavities, give the geom an explicit mass from the CAD "
          "physical-properties panel (<geom ... mass=\"0.012\"/>) instead of trusting density x volume.")


if __name__ == "__main__":
    main()
