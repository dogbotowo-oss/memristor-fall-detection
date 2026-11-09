# DFT calculations for the manuscript

Inputs, optimized structures and plot-ready data behind **Figure 3(a-e)** of the
manuscript (electronic structure of the oxygen-vacancy model and the
oxygen-vacancy migration path), plus the derived numbers quoted in the text and
in Supplementary Note S5.

All calculations were done with **Quantum ESPRESSO 7.3** (plane-wave, PAW,
PBE + Hubbard U). Nothing here needs to be re-run to reproduce the figures:
`03_data/` already holds the numerical curves, and `05_scripts/` re-makes the
plots from them.

## Model

| item | value |
|---|---|
| composition | ordered spinel ferrite, Zn0.5Ni0.25Cu0.25Fe2O4 (NiCuZnCoMnFe-O film) |
| supercell | cubic, a = 8.400 A, space group P1 (all cations explicitly ordered) |
| bulk cell | 56 atoms |
| vacancy cell | 55 atoms (one oxygen removed) |
| A sites (tetrahedral) | 4 Zn + 4 Fe (labelled FeA) |
| B sites (octahedral) | 12 Fe (labelled FeB) + 2 Ni + 2 Cu |
| atom order in .xyz/.xsf | Ni, Ni, FeB x12, Cu x2, Zn x4, FeA x4, O x31 |

In the plain `.xyz`/`.xsf` files both FeA and FeB are written as `Fe`; the
distinction is carried by the `site_labels_*.csv` tables (labels such as
`FeA21`, `FeB8`). The two B-site Fe nearest to the vacancy are **FeB#8 and
FeB#12**; they are the "vacancy-near FeB" of the d-PDOS and Loewdin panels.

## Computational settings

| item | value |
|---|---|
| code | Quantum ESPRESSO 7.3, PAW, PBE |
| Hubbard U | Fe(4.0 eV), Ni(6.0 eV), Cu(4.0 eV) |
| cutoffs | ecutwfc / ecutrho = 80 / 640 Ry |
| k-points | 2x2x2 for SCF, relax and NEB; 3x3x3 for DOS |
| smearing | Gaussian, 0.05 Ry |
| relaxation | BFGS; both bulk and vacancy cells reached `bfgs converged` (largest force component in the vacancy cell: 0.0249 eV/A) |
| migration | 5-image CI-NEB, 18 iterations, path threshold 0.05 eV/A, largest perpendicular force on the movable images 0.0447 eV/A |

## Key numbers (reproduced by `03_data/`)

Formation energies of the three candidate oxygen sites, `E = F + TS`
convention, with the updated k2 relaxation for O27:

| site | 0 K (eV) | 773 K, 1 Pa (eV) | 923 K, 1e-2 Pa (eV) |
|---|---|---|---|
| O27 | 2.328 | 1.126 | 0.684 |
| O43 | 2.800 | 1.599 | 1.156 |
| O41 | 3.125 | 1.923 | 1.481 |

O43 - O27 = +0.472 eV and O41 - O27 = +0.797 eV (spread about 0.80 eV).

Electronic structure (0.02 Ry ~ 0.272 eV Gaussian broadening, same conditions
for both cells):

- total DOS at E_F: 0.28947 (bulk) -> 1.21881 states/cell (O27 vacancy)
- Löwdin d occupation change (not a net atomic charge): all 12 FeB together
  +0.8775 e; vacancy-near FeB#8 / FeB#12 +0.4577 / +0.4600 e; the two Cu
  together +0.0339 e; Cu coordination 6 -> 5
- Fe magnetic moments about 3.84 -> 3.21-3.22 mu_B for the two vacancy-near FeB

Migration (k2 path, 80/640 Ry, 2x2x2, five images):

| convention | forward (eV) | reverse (eV) |
|---|---|---|
| F | 1.6296 | 1.5682 |
| E = F + TS | 1.6035 | 1.4719 |

The manuscript quotes "about 1.60 / 1.47 eV", i.e. the `E = F + TS` values.
Do not mix the two conventions. The moving oxygen travels 2.688 A; the vacancy
goes from site O32 to site O25, i.e. the migrating oxygen occupies the O32
vacancy and leaves a vacancy at O25.

`03_data/data_validation.json` lists every number above with its source file,
so a reader can check a quoted value without re-parsing the QE outputs.

## Layout

```
dft/
  01_inputs/       QE input files of the reported runs
  02_structures/   relaxed cells, vacancy cells, NEB images, local clusters
  03_data/         plot-ready CSV/JSON + raw dos.x/pdos output
  04_neb/          k2 NEB energy profile, geometry trajectory, barriers
  05_scripts/      parsing and plotting scripts (Python 3, numpy/matplotlib)
  06_figures/      Figure 3a-e sources (PDF/SVG/PNG) and previews
```

`MANIFEST.csv` lists every file with its size, SHA-256 and a short description.

## Not included

Raw QE outputs (`.out` of the relax/NEB jobs, per-atom
`pdos_atm#...` files, `.save`/`.wfc` wavefunctions; about 340 MB) are too large
to be tracked here. They are available from the authors on request.
