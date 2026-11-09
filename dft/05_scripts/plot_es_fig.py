# -*- coding: utf-8 -*-
"""Main figure for O27 (oxygen vacancy) local electronic response, final geometries.
(a) FeB 3d PDOS: vacancy-near FeB (atoms 8,12) vs far FeB (other 10) vs bulk;
(b) Cu 3d PDOS: O27 Cu (15,16) vs bulk Cu.
SI figure: total DOS from dos.x (same nscf, degauss 0.02 Ry = 0.272 eV).
Data: review_20261030_raw (nscf k 3x3x3, nbnd 300, degauss 0.02 Ry).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKG = r"C:\Users\lin\Desktop\论文\DFT_数据\dft压缩"
O27 = os.path.join(PKG, "09_原始输出", "review_20261030_raw", "O27ES")
BLK = os.path.join(PKG, "09_原始输出", "review_20261030_raw", "BULKES")
OUT = os.path.join(PKG, "03_DOS", "es_final_20261030")
os.makedirs(OUT, exist_ok=True)

EF_O27 = 10.514
EF_BLK = 10.500


def load_ldos(fname):
    d = np.loadtxt(fname)
    return d[:, 0], d[:, 1], d[:, 2]  # E, up, down


def find_pdos(dirpath, atom, species):
    for f in os.listdir(dirpath):
        if f"atm#{atom}({species})" in f and "(d)" in f:
            return os.path.join(dirpath, f)
    raise FileNotFoundError((dirpath, atom, species))


def avg_d(os_list, ef_shift):
    acc = None
    for f in os_list:
        e, up, dn = load_ldos(f)
        if acc is None:
            acc = np.zeros_like(e)
        acc = acc + (up + dn)
    return e - ef_shift, acc / len(os_list)


near_fe = [find_pdos(O27, a, "FeB") for a in (8, 12)]
far_fe = [find_pdos(O27, a, "FeB") for a in (3, 4, 5, 6, 7, 9, 10, 11, 13, 14)]
bulk_fe = [find_pdos(BLK, a, "FeB") for a in range(3, 15)]
o27_cu = [find_pdos(O27, a, "Cu") for a in (15, 16)]
bulk_cu = [find_pdos(BLK, a, "Cu") for a in (15, 16)]

e_n, y_n = avg_d(near_fe, EF_O27)
e_f, y_f = avg_d(far_fe, EF_O27)
e_b, y_b = avg_d(bulk_fe, EF_BLK)
e_cn, y_cn = avg_d(o27_cu, EF_O27)
e_cb, y_cb = avg_d(bulk_cu, EF_BLK)

# Lowdin d occupations (projwfc.out): near = (FeB#8 + FeB#12)/2; far = other 10 FeB
occ_near = (6.2911 + 6.2918) / 2.0
occ_far = np.mean([5.8269, 5.8282, 5.8299, 5.8227, 5.8347,
                   5.8264, 5.8288, 5.8377, 5.8406, 5.8304])
occ_bulk = 5.8350
cu_o27 = (9.3414 + 9.3456) / 2.0
cu_blk = (9.3265 + 9.3266) / 2.0

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

ax = axes[0]
ax.plot(e_b, y_b, color="0.55", lw=1.1, ls=(0, (4, 2)), label="bulk FeB")
ax.plot(e_f, y_f, color="#1f6fd0", lw=1.4, label="far FeB (O27)")
ax.plot(e_n, y_n, color="#d62728", lw=1.6, label="near FeB (O27)")
ax.axvline(0, color="k", lw=0.7, ls=":")
ax.fill_between(e_n, 0, y_n, where=(np.abs(e_n) < 0.6), color="#d62728", alpha=0.15)
ax.set_xlim(-6.5, 4.0)
ax.set_ylim(0, 4.1)
ax.set_xlabel("E - E$_F$ (eV)")
ax.set_ylabel("FeB 3d PDOS (states/eV/atom)")
ax.legend(frameon=False, fontsize=7.4, loc="upper left")
ax.set_title("(a) FeB 3d: vacancy-near vs far", fontsize=9.5)

ax = axes[1]
ax.plot(e_cb, y_cb, color="0.55", lw=1.1, ls=(0, (4, 2)), label="bulk Cu")
ax.plot(e_cn, y_cn, color="#ff7f0e", lw=1.6, label="O27 Cu (15,16)")
ax.axvline(0, color="k", lw=0.7, ls=":")
ax.set_xlim(-6.5, 4.0)
ax.set_ylim(0, 3.7)
ax.set_xlabel("E - E$_F$ (eV)")
ax.set_ylabel("Cu 3d PDOS (states/eV/atom)")
ax.legend(frameon=False, fontsize=7.4, loc="upper left")
ax.set_title("(b) Cu 3d: little d-occupation change", fontsize=9.5)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_es_near_far_Fe.{ext}"),
                dpi=400 if ext == "png" else None)
plt.close(fig)

# ---- SI figure: total DOS from dos.x ----
def load_dos(fname):
    d = np.loadtxt(fname)
    return d[:, 0], d[:, 1] + d[:, 2], d[:, 3]  # E, total DOS, integrated


e1, dos1, _ = load_dos(os.path.join(O27, "nfo_ok2r.dos"))
e2, dos2, _ = load_dos(os.path.join(BLK, "nfo_bulk.dos"))


def dos_at(e, dos, ef):
    return float(np.interp(ef, e, dos))


d_ef_o27 = dos_at(e1, dos1, EF_O27)
d_ef_blk = dos_at(e2, dos2, EF_BLK)
print("DOS(EF) O27 %.4f  bulk %.4f  ratio %.2f" % (d_ef_o27, d_ef_blk, d_ef_o27 / d_ef_blk))

fig, ax = plt.subplots(figsize=(4.2, 2.9))
ax.plot(e2 - EF_BLK, dos2, color="0.55", lw=1.1, ls=(0, (4, 2)), label="bulk")
ax.plot(e1 - EF_O27, dos1, color="#d62728", lw=1.5, label="O27 (O vacancy)")
ax.axvline(0, color="k", lw=0.7, ls=":")
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(0, 4.2)
ax.set_xlabel("E - E$_F$ (eV)")
ax.set_ylabel("Total DOS (states/eV)")
ax.text(0.98, 0.97,
        f"DOS(E$_F$): {d_ef_blk:.2f} (bulk) → {d_ef_o27:.2f} (O27), ×{d_ef_o27/d_ef_blk:.2f}\n"
        f"Gaussian smearing 0.02 Ry (0.272 eV)",
        transform=ax.transAxes, fontsize=6.9, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9))
ax.legend(frameon=True, framealpha=0.9, fontsize=7.4, loc="upper left")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_si_dos_total.{ext}"),
                dpi=400 if ext == "png" else None)
print("saved:", OUT)
