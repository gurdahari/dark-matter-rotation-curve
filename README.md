<p align="center">
  <img width="360" src="https://raw.githubusercontent.com/gurdahari/dm-assets/main/flat_curve.svg" alt="flat rotation curve">
</p>

# 🌌  Dark‑Matter Rotation‑Curve Toolkit

A clean starting point for analysing a galaxy’s rotation curve and fitting dark‑matter
halo profiles (NFW & Burkert).  Everything here is boiler‑plate; just drop your
Python code and data in the right folders and you’re ready to push.

---

## Quick‑start

```bash
# clone the repository you just created:
git clone https://github.com/<YOUR-USERNAME>/dark-matter-rotation-curve.git
cd dark-matter-rotation-curve
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run your script (example name: rotation_analysis.py)
python rotation_analysis.py -d data/Measurements/curve.csv
```

Large data files belong in **`data/Measurements/`**  
(use **Git LFS** if the folder is larger than 100 MB).

---

## Dark‑Matter 101 — one‑minute cheat‑sheet

| Topic | Key idea |
|-------|----------|
| Rotation curve | Orbital speed stays flat with radius → unseen mass. |
| Halo profile   | Formula for ρ(r); integrate → mass → predict *v(r)*. |
| NFW vs Burkert | NFW = cuspy centre; Burkert = cored. |

Further reading: `docs/reading_list.md`.

---

## Repo layout

```
dark-matter-rotation-curve/
├── rotation_analysis.py        # ← your code goes here
├── requirements.txt
├── docs/reading_list.md
├── data/
│   └── Measurements/           # ← place raw spectra here
└── .github/workflows/python.yml
```

---

## Licence

MIT — free for any use, please keep a reference to this repo.
