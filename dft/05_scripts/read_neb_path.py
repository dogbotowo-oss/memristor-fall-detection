#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse a QE neb.x path file (e.g. nfo_neb.path / nfo_neb.path0) and write
per-image geometries (Angstrom) plus image energies.

Usage:
  python read_neb_path.py PATHFILE SPECIES_POS OUTDIR

SPECIES_POS : a 55-line 'label x y z' file that defines the atom order/species
OUTDIR      : where to write img1.pos ... img5.pos and images.csv

The QE path file layout:
  RESTART INFORMATION ... NUMBER OF IMAGES ... ENERGIES, POSITIONS AND GRADIENTS
  Image:    1
       <energy Ry>
       x y z gx gy gz if_pos1 if_pos2 if_pos3     (x..z in bohr)
  ...
"""
import os
import sys

BOHR_A = 0.529177210903


def read_species(fn):
    out = []
    for ln in open(fn):
        p = ln.split()
        if len(p) == 4:
            out.append(p[0])
    return out


def main():
    path_fn, spec_fn, outdir = sys.argv[1:4]
    species = read_species(spec_fn)
    lines = open(path_fn, errors='ignore').read().splitlines()
    # locate data start
    i = 0
    for k, ln in enumerate(lines):
        if 'ENERGIES, POSITIONS AND GRADIENTS' in ln:
            i = k + 1
            break
    images = []
    cur = None
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith('Image:'):
            if cur is not None:
                images.append(cur)
            cur = {'energy': None, 'pos': []}
            i += 1
            # energy line
            if i < len(lines):
                try:
                    cur['energy'] = float(lines[i].split()[0])
                except (ValueError, IndexError):
                    cur['energy'] = None
                i += 1
            continue
        if cur is not None:
            p = s.split()
            if len(p) >= 9:
                try:
                    x, y, z = float(p[0]), float(p[1]), float(p[2])
                    cur['pos'].append((x, y, z))
                except ValueError:
                    pass
        i += 1
    if cur is not None:
        images.append(cur)
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for k, img in enumerate(images):
        pos = img['pos']
        if len(pos) == 0:
            continue
        if len(pos) != len(species):
            print('WARNING: image %d has %d atoms, expected %d' % (k + 1, len(pos), len(species)))
        fn = os.path.join(outdir, 'img%d.pos' % (k + 1))
        with open(fn, 'w') as f:
            for j, (x, y, z) in enumerate(pos):
                lab = species[j] if j < len(species) else 'X'
                f.write('%-3s %16.10f %16.10f %16.10f\n' % (lab, x * BOHR_A, y * BOHR_A, z * BOHR_A))
        rows.append((k + 1, img['energy'], len(pos), fn))
        print('image %d: E=%.8f Ry, %d atoms -> %s' % (k + 1, img['energy'], len(pos), fn))
    with open(os.path.join(outdir, 'images.csv'), 'w') as f:
        f.write('image,energy_Ry,natoms,posfile\n')
        for r in rows:
            f.write('%d,%.8f,%d,%s\n' % r)


main()
