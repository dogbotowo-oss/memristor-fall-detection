"""Plot the k2 CI-NEB barrier profile (figure 3 of the figure plan).

Reads the tight-SP recomputed numbers from sum_k2_barrier_final.json and draws
the F-convention and E = F + TS profiles.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JSON = r"C:\Users\lin\Desktop\data\sum_k2_barrier_final.json"
OUTDIR = r"C:\Users\lin\Desktop\论文\DFT_数据\dft压缩\11_结构文件_模型图_20261031"

d = json.load(open(JSON, encoding="utf-8"))
sps = d["sp"]
C_EV = 13.605693122994
F = np.array([s["F"] for s in sps])
TS = np.array([s["TS"] for s in sps])
E = np.array([s["E"] for s in sps])

relF = (F - F[0]) * C_EV
relE = (E - E[0]) * C_EV
x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

bf = d["barrier_F_fwd_eV"]; br = d["barrier_F_rev_eV"]
bef = d["barrier_E_fwd_eV"]; ber = d["barrier_E_rev_eV"]

fig, ax = plt.subplots(figsize=(6.2, 3.7))
ax.plot(x, relF, "o-", color="#c0392b", lw=2.0, ms=6, label="$F$  ($F = E - TS$, QE total energy)")
ax.plot(x, relE, "s--", color="#2c6fbb", lw=1.6, ms=5, alpha=0.85, label="$E = F + TS$  (internal energy)")

# mark climbing image
ax.plot([x[2]], [relF[2]], "o", ms=11, mfc="none", mec="#c0392b", mew=1.8)
ax.annotate("climbing image (TS)", xy=(x[2], relF[2]), xytext=(x[2] + 0.03, relF[2] + 0.10),
            fontsize=8.5, color="#c0392b")

# barrier labels, white background so they stay readable over the lines
bbox = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
ax.text(0.03, 1.36, "$E_a(\\rightarrow)$ = %.3f eV" % bf, fontsize=9.5, color="#7b241c", bbox=bbox)
ax.text(0.64, 1.27, "$E_a(\\leftarrow)$ = %.3f eV" % br, fontsize=9.5, color="#7b241c", bbox=bbox)
# final-state energy offset
ax.annotate("", xy=(1.0, relF[4]), xytext=(1.0, 0.0),
            arrowprops=dict(arrowstyle="<->", color="0.5", lw=0.9, ls=":"))
ax.text(1.015, relF[4] / 2, "%.3f eV" % relF[4], fontsize=8, color="0.35", va="center")

ax.set_xlabel("reaction coordinate  (O32 $\\rightarrow$ O25)", fontsize=10)
ax.set_ylabel("relative energy (eV)", fontsize=10)
ax.set_xlim(-0.06, 1.12)
ax.set_xticks(x)
ax.set_xticklabels(["initial\nimage 1", "image 2", "TS\nimage 3", "image 4", "final\nimage 5"], fontsize=8.5)
ax.set_ylim(-0.07, relF[2] + 0.26)
ax.grid(alpha=0.22, ls=":")
ax.legend(loc="upper left", fontsize=8.5, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
os.makedirs(OUTDIR, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUTDIR, "fig3_barrier_profile." + ext), dpi=300)
print("barrier_F_fwd = %.5f  barrier_F_rev = %.5f" % (bf, br))
print("barrier_E_fwd = %.5f  barrier_E_rev = %.5f" % (bef, ber))
print("relF:", " ".join("%.4f" % v for v in relF))
print("relE:", " ".join("%.4f" % v for v in relE))
print("wrote", os.path.join(OUTDIR, "fig3_barrier_profile.png/pdf"))
