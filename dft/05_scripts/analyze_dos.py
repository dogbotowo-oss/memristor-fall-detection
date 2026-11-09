#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse QE dos.x / projwfc.x output for the NiFe2O4 bulk DOS stage.

Usage:
  python analyze_dos.py DOS_DAT FILPDOS_PREFIX OUTDIR

DOS_DAT      : fildos of dos.x (columns: E(eV) dosup dosdown, '#' comments skipped)
FILPDOS_PREFIX: filpdos of projwfc.x, e.g. "bulk_pdos"; per-atom files are
                bulk_pdos.pdos_atm#N(Species)_wfc#k(label).dat
OUTDIR       : output directory (03_DOS)

Writes:
  OUTDIR/pdos_FeA_3d.dat, pdos_FeB_3d.dat, pdos_Ni_3d.dat, pdos_O_2p.dat
      columns: E(eV)  PDOS_up  PDOS_down
  OUTDIR/dos_summary.txt  (integrated moments, gap estimate around E_F)
"""
import sys
import os
import glob
import re

import numpy as np


def read_dos(fn):
    rows = []
    for ln in open(fn, errors='ignore'):
        s = ln.strip()
        if not s or s.startswith('#') or s.startswith('&'):
            continue
        p = s.split()
        try:
            rows.append([float(x) for x in p[:3]])
        except ValueError:
            continue
    a = np.array(rows)
    return a[:, 0], a[:, 1], (a[:, 2] if a.shape[1] > 2 else np.zeros(len(a)))


def read_pdos(fn):
    rows = []
    for ln in open(fn, errors='ignore'):
        s = ln.strip()
        if not s or s.startswith('#') or s.startswith('&'):
            continue
        p = s.split()
        try:
            rows.append([float(x) for x in p[:3]])
        except ValueError:
            continue
    if not rows:
        return None, None, None
    a = np.array(rows)
    return a[:, 0], a[:, 1], (a[:, 2] if a.shape[1] > 2 else np.zeros(len(a)))


TARGETS = {'FeA': 'd', 'FeB': 'd', 'Ni': 'd', 'O': 'p'}


def main():
    dos_fn, pref, outdir = sys.argv[1:4]
    os.makedirs(outdir, exist_ok=True)
    E, up, dn = read_dos(dos_fn)
    np.savetxt(os.path.join(outdir, 'dos.dat'), np.column_stack([E, up, dn]),
               fmt='%.6f', header='E(eV) DOS_up DOS_down', comments='')
    files = sorted(glob.glob(pref + '.pdos_atm#*_wfc#*'))
    grouped = {k: [] for k in TARGETS}
    natoms = {}
    for fn in files:
        m = re.search(r'pdos_atm#(\d+)\(([^)]+)\)_wfc#(\d+)\(([^)]+)\)', os.path.basename(fn))
        if not m:
            continue
        idx, species, iwfc, lab = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        natoms.setdefault(species, set()).add(idx)
        if species in TARGETS and lab == TARGETS[species]:
            grouped[species].append(fn)
    summary = ['QE PDOS analysis', 'total DOS: %s (%d points)' % (dos_fn, len(E))]
    for species, fns in grouped.items():
        if not fns:
            summary.append('%s: no matching pdos files' % species)
            continue
        acc_u = acc_d = None
        for fn in fns:
            e, u, d = read_pdos(fn)
            if e is None:
                continue
            if acc_u is None:
                acc_u, acc_d = u.copy(), d.copy()
            elif len(e) == len(acc_u):
                acc_u += u
                acc_d += d
        if acc_u is None:
            continue
        out = os.path.join(outdir, 'pdos_%s_3d.dat' % species) if TARGETS[species] == 'd' \
            else os.path.join(outdir, 'pdos_%s_2p.dat' % species)
        np.savetxt(out, np.column_stack([E, acc_u, acc_d]), fmt='%.6f',
                   header='E(eV) PDOS_up PDOS_down', comments='')
        # d-electron count estimate around the occupied window
        summary.append('%s: n_atoms=%d files=%d  -> %s' % (species, len(natoms.get(species, [])), len(fns), out))
    # crude gap estimate: energies where total DOS falls below 1% of its maximum
    tot = up + dn
    if tot.max() > 0:
        thr = 0.01 * tot.max()
        low = [i for i in range(len(tot)) if tot[i] < thr]
        summary.append('near-zero-DOS window: energy range of points with DOS<1%% max (indices): %s' %
                       (str(low[:5]) + '...' if len(low) > 5 else str(low)))
    with open(os.path.join(outdir, 'dos_summary.txt'), 'w') as f:
        f.write('\n'.join(summary) + '\n')
    print('\n'.join(summary))


main()
