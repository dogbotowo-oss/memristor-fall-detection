#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Map two relaxed vacancy endpoints onto a COMMON 55-atom order for NEB.

Geometry conventions
--------------------
BULK      : 56-atom relaxed structure (labels x y z, angstrom)
i (1-based): vacancy site of endpoint A  (missing O in A_final)
j (1-based): vacancy site of endpoint B  (missing O in B_final); j is a
             nearest-neighbour O of i. The migrating O goes from site j (in A)
             to site i (in B).
A_final   : 55 atoms, order = BULK order with atom i removed
B_final   : 55 atoms, order = BULK order with atom j removed

Common order used for NEB = BULK order with atom i removed (so it matches
A_final 1:1).  In that order the migrating O occupies the slot of atom j; for
endpoint B its coordinates are taken from the position of atom i in B_final.

Usage:
  python3 map_neb_endpoints.py BULK.ang I J A_final.ang B_final.ang OUTDIR
Writes A_common.ang, B_common.ang, mapping.csv and prints the displacement
check (only the migrating atom may move by more than ~0.1 A).
"""
import os
import sys


def read_pos(fn):
    return [[p[0], float(p[1]), float(p[2]), float(p[3])]
            for p in (l.split() for l in open(fn)) if len(p) == 4]


def write_pos(pos, fn):
    with open(fn, 'w') as f:
        for p in pos:
            f.write('%-3s %16.10f %16.10f %16.10f\n' % (p[0], p[1], p[2], p[3]))


def main():
    bulk_fn, i, j, afn, bfn, outdir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
    bulk = read_pos(bulk_fn)
    A = read_pos(afn)
    B = read_pos(bfn)
    nat = len(bulk)
    assert len(A) == nat - 1 and len(B) == nat - 1, 'endpoint files must have nat-1 atoms'
    assert bulk[i - 1][0] == 'O' and bulk[j - 1][0] == 'O', 'sites i,j must be O'
    os.makedirs(outdir, exist_ok=True)

    def a_index(k):           # 1-based index of bulk atom k inside A_final
        return k - (1 if k > i else 0)

    def b_index(k):           # 1-based index of bulk atom k inside B_final
        return k - (1 if k > j else 0)

    # sanity: species ordering check
    for k in range(1, nat + 1):
        if k == i:
            continue
        assert A[a_index(k) - 1][0] == bulk[k - 1][0], 'A species mismatch at bulk atom %d' % k
    for k in range(1, nat + 1):
        if k == j:
            continue
        assert B[b_index(k) - 1][0] == bulk[k - 1][0], 'B species mismatch at bulk atom %d' % k

    A_common, B_common, rows = [], [], []
    slot = 0
    for k in range(1, nat + 1):
        if k == i:
            continue
        slot += 1
        apos = A[a_index(k) - 1]
        if k == j:
            # migrating O: in B it sits at site i -> take B coordinates of bulk atom i
            bpos = B[b_index(i) - 1]
        else:
            bpos = B[b_index(k) - 1]
        A_common.append([apos[0], apos[1], apos[2], apos[3]])
        B_common.append([bpos[0], bpos[1], bpos[2], bpos[3]])
        rows.append((slot, k, apos[0], apos[1], apos[2], apos[3], bpos[1], bpos[2], bpos[3]))

    write_pos(A_common, os.path.join(outdir, 'A_common.ang'))
    write_pos(B_common, os.path.join(outdir, 'B_common.ang'))
    with open(os.path.join(outdir, 'mapping.csv'), 'w') as f:
        f.write('slot,bulk_index,label,A_x,A_y,A_z,B_x,B_y,B_z\n')
        for r in rows:
            f.write('%d,%d,%s,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n' % r)

    # displacement check (minimum image, cubic 8.4 A)
    def mic(d):
        return (d + 0.5 * 8.4) % 8.4 - 0.5 * 8.4

    def dist(a, b):
        return (mic(a[1] - b[1]) ** 2 + mic(a[2] - b[2]) ** 2 + mic(a[3] - b[3]) ** 2) ** 0.5

    disp = [(slot + 1, k, dist(A_common[slot], B_common[slot]))
            for slot, k in enumerate([k for k in range(1, nat + 1) if k != i])]
    mig_slot = [s for s, k in enumerate([k for k in range(1, nat + 1) if k != i], start=1) if k == j][0]
    print('migrating O: bulk atom %d (site j) -> occupies slot %d (1-based) in both endpoints' % (j, mig_slot))
    print('A coordinates of migrating O: %.4f %.4f %.4f' % tuple(A_common[mig_slot - 1][1:]))
    print('B coordinates of migrating O: %.4f %.4f %.4f' % tuple(B_common[mig_slot - 1][1:]))
    ordered = sorted(disp, key=lambda r: -r[2])
    print('\nlargest per-atom displacements A->B:')
    for slot, k, d in ordered[:5]:
        print('  slot %2d (bulk atom %2d): %.4f A' % (slot, k, d))
    others = [d for slot, k, d in disp if k != j]
    print('\ncheck: max |dR| for the 54 non-migrating atoms = %.4f A (must be < 0.1)' % max(others))
    print('check: migrating atom displacement = %.4f A (should be ~2.7)' % [d for s, k, d in disp if k == j][0])


main()
