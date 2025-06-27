<p align="center">
  <img width="360" src="docs/figures/rotation_curve_banner.jpg" alt="flat rotation curve">
</p>

# 🌌  Dark‑Matter Rotation‑Curve Toolkit

A fully reproducible pipeline for converting 21 cm H I spectra into the Milky‑Way
rotation curve and fitting two canonical dark‑matter halo profiles
(**NFW** & **Burkert**) alongside baryonic components
(**Hernquist bulge** + **Sérsic disc**).

---

## 1 Quick start

```bash
git clone https://github.com/<YOUR-USERNAME>/dark-matter-rotation-curve.git
cd dark-matter-rotation-curve
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# analyse every CSV in data/Measurements/
python rotation_analysis.py -d data/Measurements
```

> Data folders larger than **100 MB** should be tracked with **Git LFS**.

---

## 2 What the script does

| Stage | Purpose |
|-------|---------|
| **Ingest spectra** | reads each `*.csv`, extracts brightness‑temperature vs radial velocity. |
| **Peak finding** | `scipy.signal.find_peaks` → terminal velocity for each longitude. |
| **Coordinate transform** | converts (l, v) → (*R*, *vₜ*), propagates uncertainties. |
| **Quadrants I & IV curves** | builds two curves, computes a weighted average. |
| **Polynomial & piece‑wise fits** | cubic over full range + linear inside the bulge. |
| **Mass profile** | computes enclosed mass *M(<R)* and its uncertainty. |
| **Halo fits** | fits NFW & Burkert + baryonic terms; plots theory vs data. |
| **Total‑mass integral** | integrates analytic density laws to 200 kpc. |

All figures are saved to `docs/figures/` and displayed interactively.

---

## 3 Directory layout

```
dark-matter-rotation-curve/
├── rotation_analysis.py        ← analysis script
├── requirements.txt
├── docs/
│   ├── reading_list.md
│   └── figures/                ← generated plots
├── data/
│   └── Measurements/           ← raw 21 cm spectra or CSV curve
└── .github/workflows/python.yml
```

---

## 4 Scientific background (brief)

The Milky Way’s rotation curve is nearly flat beyond the optical disc,
inconsistent with the √1/r decline expected from visible matter alone.
Analytic halo profiles show that ≈ 5 × 10¹¹ M<sub>⊙</sub> of dark matter within
200 kpc reproduces the data; the baryon‑to‑DM mass ratio at *R*<sub>0</sub> is
≈ 0.17, in line with ΛCDM.

---

## 5 Key references

* **Navarro, Frenk & White (1997)** – universal CDM halo profile  
* **Burkert (1995)** – cored empirical halo  
* **Sofue (2020)** – Milky‑Way mass & rotation‑curve review  
* **Bullock & Boylan‑Kolchin (2017)** – small‑scale challenges to ΛCDM  

Further reading in `docs/reading_list.md`.

---

## Licence

**MIT** – free for academic or commercial use; please cite this repository.

<p align="center">Made with ☕ and <em>cold dark matter</em> (ΛCDM)</p>
