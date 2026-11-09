# -*- coding: utf-8 -*-
"""Generate structure files (XYZ/XSF/CIF) for the model figure.

Sources:
  bulk final relaxed positions : jobs_20261030/_geom/bulk_final_positions.txt
  O27  final relaxed positions : jobs_20261030/_geom/o27_final_positions.txt
  k2 endpoints (relaxed)       : 06_p1_native_relax/ep_k2/ep1k2/e1k2.out
                                06_p1_native_relax/ep_k2/ep5k2b/e5kb.out
  k1 converged NEB path images : 09_原始输出/review_20261030_raw/neb_adapt5_path/nfo_nebMD7.path
  labels for path/endpoints    : 02_NEB收敛/neb_adapt5_20261030/neb_adapt5.in

Output: 11_结构文件_模型图_20261031/
"""
import os, sys, math

PKG = r"C:\Users\lin\Desktop\论文\DFT_数据\dft压缩"
DATA = r"C:\Users\lin\Desktop\data"
GEOM = os.path.join(DATA, "jobs_20261030", "_geom")
OUT = os.path.join(PKG, "11_结构文件_模型图_20261031")
BOX = 8.4
BOHR = 0.529177210903
TYPES = ["Zn", "FeA", "FeB", "Ni", "Cu", "O"]
ELEM = {"Zn": "Zn", "FeA": "Fe", "FeB": "Fe", "Ni": "Ni", "Cu": "Cu", "O": "O"}
ROLE = {"Zn": "tetrahedral A-site", "FeA": "tetrahedral A-site",
        "FeB": "octahedral B-site", "Ni": "octahedral B-site",
        "Cu": "octahedral B-site", "O": "anion"}


def read_positions_txt(fn):
    atoms = []
    for ln in open(fn, encoding="utf-8"):
        p = ln.split()
        if len(p) < 4:
            continue
        atoms.append((p[0], [float(p[1]), float(p[2]), float(p[3])]))
    return atoms


def parse_positions_block(lines, start):
    atoms = []
    i = start + 1
    while i < len(lines):
        p = lines[i].split()
        if len(p) < 4:
            break
        try:
            xyz = [float(p[1]), float(p[2]), float(p[3])]
        except ValueError:
            break
        atoms.append((p[0], xyz))
        i += 1
    return atoms


def parse_neb5(fn):
    lines = open(fn, encoding="utf-8", errors="ignore").read().split("\n")
    images = []
    cur = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("BEGIN_POSITIONS"):
            images = []
            cur = None
        elif s in ("FIRST_IMAGE", "INTERMEDIATE_IMAGE", "LAST_IMAGE"):
            cur = []
            images.append(cur)
        elif s.startswith("ATOMIC_POSITIONS"):
            atoms = parse_positions_block(lines, i)
            if cur is not None and not cur:
                cur.extend(atoms)
    return images


def parse_final_positions(fn):
    lines = open(fn, encoding="utf-8", errors="ignore").read().split("\n")
    idx = None
    for i, ln in enumerate(lines):
        if "Begin final coordinates" in ln:
            idx = i
    if idx is None:
        for i, ln in enumerate(lines):
            if ln.strip().startswith("ATOMIC_POSITIONS"):
                idx = i - 1
    res = None
    i = idx + 1 if idx is not None else 0
    for j in range(i, len(lines)):
        if lines[j].strip().startswith("ATOMIC_POSITIONS") and len(lines[j].split()) <= 3:
            atoms = parse_positions_block(lines, j)
            if len(atoms) >= 50:
                res = atoms
    return res


def parse_path_final(fn):
    lines = open(fn, encoding="utf-8", errors="ignore").read().split("\n")
    starts = [i for i, ln in enumerate(lines) if ln.strip().startswith("Image:")]
    final = starts[-5:]
    if len(final) != 5:
        return None
    images = []
    for s in final:
        if lines[s + 1].strip() not in ("T", "F"):
            return None
        atoms, j = [], s + 2
        while j < len(lines):
            p = lines[j].split()
            if len(p) != 3:
                break
            try:
                atoms.append([float(p[0]) * BOHR, float(p[1]) * BOHR, float(p[2]) * BOHR])
            except ValueError:
                break
            j += 1
        if len(atoms) < 50:
            return None
        images.append(atoms)
    return images


def unwrap(xyz, ref):
    best, out = None, None
    for nx in (-1, 0, 1):
        for ny in (-1, 0, 1):
            for nz in (-1, 0, 1):
                cand = [xyz[0] + nx * BOX, xyz[1] + ny * BOX, xyz[2] + nz * BOX]
                d = sum((cand[k] - ref[k]) ** 2 for k in range(3))
                if best is None or d < best:
                    best, out = d, cand
    return out


def min_image(dv):
    out = []
    for k in range(3):
        v = dv[k] % BOX
        if v > BOX / 2:
            v -= BOX
        out.append(v)
    return out


def wrap(pos):
    return [[x % BOX, y % BOX, z % BOX] for (x, y, z) in pos]


def write_xyz(fn, labels, pos, comment):
    with open(fn, "w", encoding="utf-8", newline="\n") as f:
        f.write("%d\n%s\n" % (len(labels), comment))
        for lab, (x, y, z) in zip(labels, pos):
            f.write("%-2s %14.8f %14.8f %14.8f\n" % (ELEM[lab], x, y, z))


def write_xsf(fn, labels, pos, comment):
    with open(fn, "w", encoding="utf-8", newline="\n") as f:
        f.write("# %s\nCRYSTAL\nPRIMVEC\n" % comment)
        f.write("%12.6f %12.6f %12.6f\n" % (BOX, 0, 0))
        f.write("%12.6f %12.6f %12.6f\n" % (0, BOX, 0))
        f.write("%12.6f %12.6f %12.6f\n" % (0, 0, BOX))
        f.write("PRIMCOORD\n%d 1\n" % len(labels))
        for lab, (x, y, z) in zip(labels, pos):
            f.write("%-2s %14.8f %14.8f %14.8f\n" % (ELEM[lab], x, y, z))


def write_cif(fn, labels, pos, comment, vacancy=None):
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    rows = []
    for i, (lab, (x, y, z)) in enumerate(zip(labels, pos), 1):
        rows.append("  %-6s %-3s %10.7f %10.7f %10.7f 1.0000" %
                    ("%s%d" % (lab, i), ELEM[lab], x / BOX, y / BOX, z / BOX))
    with open(fn, "w", encoding="utf-8", newline="\n") as f:
        f.write("# %s\n" % comment)
        f.write("data_%s\n" % "structure")
        f.write("_cell_length_a %.6f\n_cell_length_b %.6f\n_cell_length_c %.6f\n" % (BOX, BOX, BOX))
        f.write("_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n")
        f.write("_symmetry_space_group_name_H-M 'P 1'\n")
        f.write("_symmetry_Int_Tables_number 1\n")
        f.write("loop_\n_symmetry_equiv_pos_as_xyz\n  'x, y, z'\n")
        f.write("loop_\n_atom_site_label\n_atom_site_type_symbol\n")
        f.write("_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n")
        f.write("_atom_site_occupancy\n")
        for r in rows:
            f.write(r + "\n")
        # vacancy marker as a zero-occupancy site
        if vacancy is not None:
            f.write("  %-6s %-3s %10.7f %10.7f %10.7f 0.0000\n" %
                    ("Vac", "He", vacancy[0] / BOX, vacancy[1] / BOX, vacancy[2] / BOX))


def write_site_labels(fn, rows):
    with open(fn, "w", encoding="utf-8", newline="\n") as f:
        f.write("index,file_element,dft_type,role,is_near_vacancy,is_vacancy_site,x_ang,y_ang,z_ang\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    bulk = read_positions_txt(os.path.join(GEOM, "bulk_final_positions.txt"))
    o27 = read_positions_txt(os.path.join(GEOM, "o27_final_positions.txt"))
    assert len(bulk) == 56 and len(o27) == 55, (len(bulk), len(o27))

    # path / endpoint labels come from neb_adapt5.in (Ni, Ni, FeB x12, Cu x2, Zn x4, FeA x4, O x31)
    neb5_in = os.path.join(PKG, "02_NEB收敛", "neb_adapt5_20261030", "neb_adapt5.in")
    imgs = parse_neb5(neb5_in)
    labels_path = [a[0] for a in imgs[0]]
    assert labels_path[:2] == ["Ni", "Ni"] and labels_path[14] == "Cu", labels_path[:16]
    assert len(labels_path) == 55

    e1 = parse_final_positions(os.path.join(PKG, "06_p1_native_relax", "ep_k2", "ep1k2", "e1k2.out"))
    e5 = parse_final_positions(os.path.join(PKG, "06_p1_native_relax", "ep_k2", "ep5k2b", "e5kb.out"))
    assert e1 and e5 and len(e1) == len(e5) == 55
    e1pos = [unwrap(xyz, imgs[0][j][1]) for j, (lab, xyz) in enumerate(e1)]
    e5pos = [unwrap(xyz, imgs[4][j][1]) for j, (lab, xyz) in enumerate(e5)]

    path_file = os.path.join(PKG, "09_原始输出", "review_20261030_raw",
                             "neb_adapt5_path", "nfo_nebMD7.path")
    pimgs = parse_path_final(path_file)
    assert pimgs and len(pimgs) == 5 and len(pimgs[0]) == 55

    # bulk: first 26 atoms are same order as O27; bulk atom 27 (1-based) is the removed O
    bulk_labels = [a[0] for a in bulk]
    vac = bulk[26][1]
    assert bulk_labels[26] == "O", bulk_labels[26]

    # ---------- bulk ----------
    write_xyz(os.path.join(OUT, "bulk_relaxed.xyz"), bulk_labels, wrap([a[1] for a in bulk]),
              "bulk P1 cell, 56 atoms, 80/640 Ry k2 native relax (F=-8859.0249898555 Ry)")
    write_xsf(os.path.join(OUT, "bulk_relaxed.xsf"), bulk_labels, wrap([a[1] for a in bulk]),
              "bulk relaxed 56 atoms, a=8.4 A")
    write_cif(os.path.join(OUT, "bulk_relaxed.cif"), bulk_labels, wrap([a[1] for a in bulk]),
              "bulk relaxed 56 atoms, a=8.4 A")

    # ---------- O27 vacancy ----------
    o27_labels = [a[0] for a in o27]
    write_xyz(os.path.join(OUT, "O27_vacancy_relaxed.xyz"), o27_labels, wrap([a[1] for a in o27]),
              "O27 oxygen vacancy, 55 atoms, k2 native relax 223767 (F=-8817.33913441 Ry)")
    write_xsf(os.path.join(OUT, "O27_vacancy_relaxed.xsf"), o27_labels, wrap([a[1] for a in o27]),
              "O27 vacancy relaxed 55 atoms, a=8.4 A")
    write_cif(os.path.join(OUT, "O27_vacancy_relaxed.cif"), o27_labels, wrap([a[1] for a in o27]),
              "O27 vacancy relaxed 55 atoms", vacancy=vac)

    # marked xyz: 55 atoms + dummy vacancy atom placed at the removed bulk O site
    with open(os.path.join(OUT, "O27_vacancy_marked.xyz"), "w", encoding="utf-8", newline="\n") as f:
        f.write("%d\n%s\n" % (56, "O27 vacancy relaxed + X marker at the removed O site"))
        for lab, (x, y, z) in zip(o27_labels, wrap([a[1] for a in o27])):
            f.write("%-2s %14.8f %14.8f %14.8f\n" % (ELEM[lab], x, y, z))
        f.write("%-2s %14.8f %14.8f %14.8f\n" % ("X", vac[0], vac[1], vac[2]))

    # ---------- k2 endpoints ----------
    write_xyz(os.path.join(OUT, "ep1_reactant_k2_relaxed.xyz"), labels_path, wrap(e1pos),
              "reactant endpoint, 80/640 Ry k2 native relax 223864 (F=-8817.29509330 Ry)")
    write_xsf(os.path.join(OUT, "ep1_reactant_k2_relaxed.xsf"), labels_path, wrap(e1pos),
              "reactant endpoint k2 relaxed")
    write_xyz(os.path.join(OUT, "ep5_product_k2_relaxed.xyz"), labels_path, wrap(e5pos),
              "product endpoint, 80/640 Ry k2 native relax 224049 (F=-8817.29058242 Ry)")
    write_xsf(os.path.join(OUT, "ep5_product_k2_relaxed.xsf"), labels_path, wrap(e5pos),
              "product endpoint k2 relaxed")

    # ---------- k1 converged path images ----------
    pdir = os.path.join(OUT, "path_k1_adapt5")
    os.makedirs(pdir, exist_ok=True)
    for k in range(5):
        write_xyz(os.path.join(pdir, "img%d.xyz" % (k + 1)), labels_path, wrap(pimgs[k]),
                  "CI-NEB k1 converged path image %d (neb_adapt5 final .path)" % (k + 1))
    # copy the native multi-image files (VESTA/OVITO can animate the .axsf)
    import shutil
    for src in ("nfo_nebMD7.xyz", "nfo_nebMD7.axsf", "nfo_nebMD7.crd", "nfo_nebMD7.dat"):
        s = os.path.join(PKG, "09_原始输出", "review_20261030_raw", "neb_adapt5_path", src)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(pdir, src))

    # ---------- site labels / near-vacancy neighbours ----------
    cat = []  # (index, elem, type, role, near, vacancy, x,y,z)
    for i, (lab, xyz) in enumerate(o27, 1):
        cat.append((i, ELEM[lab], lab, ROLE[lab], "", "", xyz[0], xyz[1], xyz[2]))
    # neighbours of the vacancy (cation within 2.6 A of vac site)
    neigh = []
    for i, (lab, xyz) in enumerate(o27, 1):
        if lab == "O":
            continue
        d = math.sqrt(sum(v * v for v in min_image([xyz[k] - vac[k] for k in range(3)])))
        if d < 2.9:
            neigh.append((i, lab, d))
    near_idx = sorted(i for (i, lab, d) in neigh)
    rows = []
    for i, (lab, xyz) in enumerate(o27, 1):
        rows.append((i, ELEM[lab], lab, ROLE[lab], "yes" if i in near_idx else "no", "", xyz[0], xyz[1], xyz[2]))
    write_site_labels(os.path.join(OUT, "site_labels_O27.csv"), rows)
    rows_b = []
    for i, (lab, xyz) in enumerate(bulk, 1):
        rows_b.append((i, ELEM[lab], lab, ROLE[lab], "", "yes" if i == 27 else "", xyz[0], xyz[1], xyz[2]))
    write_site_labels(os.path.join(OUT, "site_labels_bulk.csv"), rows_b)

    print("wrote files to", OUT)
    print("vacancy site (bulk O27) = %.6f %.6f %.6f A  (frac %.5f %.5f %.5f)" %
          (vac[0], vac[1], vac[2], vac[0] / BOX, vac[1] / BOX, vac[2] / BOX))
    print("cation neighbours of vacancy (<2.9 A):")
    for i, lab, d in neigh:
        print("  atom %2d  %-4s  d=%.4f A" % (i, lab, d))


main()
