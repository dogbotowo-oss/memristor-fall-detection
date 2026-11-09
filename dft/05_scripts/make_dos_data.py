#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shift DOS/PDOS energy axes to E - E_F and write plotting data.

Usage: python make_dos_data.py DOS_DIR
Reads dos.dat (header gives EFermi) and pdos_*.dat, writes DOS_DIR/shifted/*.dat
with columns:  E-EF(eV)  up  down
"""
import os
import re
import sys


def load(fn):
    rows = []
    for ln in open(fn, errors='ignore'):
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        p = s.split()
        if len(p) >= 3:
            try:
                rows.append((float(p[0]), float(p[1]), float(p[2])))
            except ValueError:
                continue
    return rows


def main():
    d = sys.argv[1]
    header = open(os.path.join(d, 'dos.dat')).readline()
    m = re.search(r'EFermi\s*=\s*([-\d.]+)', header)
    ef = float(m.group(1))
    out = os.path.join(d, 'shifted')
    os.makedirs(out, exist_ok=True)
    for name in ('dos', 'pdos_FeA_3d', 'pdos_FeB_3d', 'pdos_Ni_3d', 'pdos_O_2p', 'pdos_tot'):
        fn = os.path.join(d, name + '.dat')
        if not os.path.exists(fn):
            continue
        rows = load(fn)
        with open(os.path.join(out, name + '_shift.dat'), 'w') as f:
            f.write('# E-EF(eV) up down  [EF=%.4f eV]\n' % ef)
            for e, u, dn in rows:
                f.write('%.5f %.8e %.8e\n' % (e - ef, u, dn))
        print('%s -> shifted (%d rows)' % (name, len(rows)))
    print('E_F = %.4f eV' % ef)


main()
