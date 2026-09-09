"""
Export a minimal reviewer-facing reproduction package from MASTER.

Destination (default): ../2026_FPD_Ales_Padro_Bajo_Nervion

Includes:
  - definitive Bajo Nervión PM10 CSV (zipped; public data)
  - EDA / ACF / feature-selection artefacts
  - code needed to rebuild packs, train, and regenerate paper tables/figures
  - configs for L in {24,48,72,96} and supplementary H=48
  - light results tables + manuscript figures (v02)
Excludes: archives, raw dumps, checkpoints, other zones, notebooks clutter, venv.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_NAME = "2026_FPD_Ales_Padro_Bajo_Nervion"
DEFAULT_DEST = ROOT.parent / REPO_NAME
GITHUB_URL = f"https://github.com/apadroh/{REPO_NAME}"

# (src relative to ROOT, dest relative to package) — files or dirs
COPY_PATHS: list[tuple[str, str]] = [
    # --- core code ---
    ("main.py", "main.py"),
    ("requirements.txt", "requirements.txt"),
    ("src/utils", "src/utils"),
    ("src/data", "src/data"),
    ("src/graphs", "src/graphs"),
    ("src/experiments", "src/experiments"),
    ("src/models/informer2020", "src/models/informer2020"),
    ("src/models/airformer", "src/models/airformer"),
    ("src/models/gat_informer", "src/models/gat_informer"),
    # --- analysis (EDA / ACF / comparison used in TFM) ---
    ("analysis/README.md", "analysis/README.md"),
    ("analysis/eda", "analysis/eda"),
    ("analysis/acf", "analysis/acf"),
    ("analysis/feature_selection/zone_2/PM10", "analysis/feature_selection/zone_2/PM10"),
    ("analysis/comparison", "analysis/comparison"),
    ("scripts/analysis", "scripts/analysis"),
    ("scripts/hpc", "scripts/hpc"),
    # --- model notebook entrypoints (thin) ---
    ("models/informer2020/README.md", "models/informer2020/README.md"),
    ("models/airformer/README.md", "models/airformer/README.md"),
    ("models/gat_informer/README.md", "models/gat_informer/README.md"),
    # --- light results (tables + paper figures) ---
    ("results/comparison/zone_2/all_models/tables", "results/comparison/zone_2/all_models/tables"),
    ("results/comparison/zone_2/seq_len_sensitivity", "results/comparison/zone_2/seq_len_sensitivity"),
    ("results/comparison/zone_2/horizon_48", "results/comparison/zone_2/horizon_48"),
    ("results/README.md", "results/README.md"),
    ("docs/figures_print", "docs/figures_print"),
    ("docs/tfm_manuscript_overleaf.tex", "docs/tfm_manuscript_overleaf.tex"),
    ("docs/tfm_manuscript_overleaf_v02.tex", "docs/tfm_manuscript_overleaf_v02.tex"),
    ("docs/biblio.bib", "docs/biblio.bib"),
    # --- dartboard / graph artefacts (tiny) ---
    (
        "src/models/airformer/_upstream/data/local_partition/zone2_18",
        "data/graphs/zone2_18_dartboard",
    ),
]

SKIP_DIR_NAMES = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".git",
    "checkpoints",
    "AIR_TINY",
    "50",
    "50-200",
    "50-200-500",
    "25-100-250",
    "logs",
}
SKIP_SUFFIXES = {".pth", ".pt", ".ckpt", ".pyc"}
SKIP_NAMES = {"data.zip", ".env"}


def should_skip(path: Path) -> bool:
    if path.name in SKIP_NAMES or path.name in SKIP_DIR_NAMES:
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    parts = set(path.parts)
    if "local_partition" in parts and path.name not in {
        "zone2_18",
        "assignment.npy",
        "mask.npy",
        "README.txt",
    }:
        if "zone2_18" not in parts and path.is_dir():
            if path.parent.name == "local_partition":
                return True
    return False


def copy_tree(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    for p in src.rglob("*"):
        if should_skip(p):
            continue
        if any(part in SKIP_DIR_NAMES for part in p.relative_to(src).parts):
            continue
        rel = p.relative_to(src)
        target = dst / rel
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            if p.suffix.lower() in SKIP_SUFFIXES or p.name in SKIP_NAMES:
                continue
            if p.suffix == ".npy" and p.stat().st_size > 50_000_000:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)


def zip_definitive_csv(dest_root: Path) -> Path:
    csv = ROOT / "data" / "training" / "zone_2" / "dataset_zone_2_PM10.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    out_dir = dest_root / "data" / "training" / "zone_2"
    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = out_dir / "dataset_zone_2_PM10.csv.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(csv, arcname=csv.name)
    man = ROOT / "data" / "training" / "zone_2" / "prepare_training_manifest_PM10.json"
    if man.exists():
        shutil.copy2(man, out_dir / man.name)
    return zpath


def copy_xgboost_summaries(dest_root: Path) -> None:
    """Light XGBoost / Informer / AirFormer batch summaries (no preds)."""
    patterns = [
        ROOT.glob("results/xgboost/zone_2/batch_summary_*.csv"),
        ROOT.glob("results/xgboost/zone_2/seq_len_*/batch_summary_*.csv"),
        ROOT.glob("results/xgboost/zone_2/horizon_48/batch_summary_*.csv"),
        ROOT.glob("results/informer2020/zone_2/horizon_48/batch_summary_*.csv"),
        ROOT.glob("results/comparison/zone_2/all_models/*.md"),
        ROOT.glob("results/comparison/zone_2/all_models/*.json"),
    ]
    for it in patterns:
        for src in it:
            rel = src.relative_to(ROOT)
            dst = dest_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  copy {rel}")


def copy_airformer_mse_summaries(dest_root: Path) -> None:
    """AirFormer MSE loss sensitivity (results.json + summaries only)."""
    base = ROOT / "results" / "airformer" / "zone_2" / "ZONE2_PM10"
    if not base.exists():
        return
    for d in sorted(base.glob("*_mse")):
        if not d.is_dir():
            continue
        for src in d.rglob("*"):
            if src.is_dir():
                continue
            if src.suffix.lower() in {".pth", ".pt", ".ckpt", ".npy"}:
                continue
            if "checkpoints" in src.parts or src.parent.name == "logs":
                continue
            rel = src.relative_to(ROOT)
            dst = dest_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  copy {rel}")


def write_configs(dest_root: Path) -> None:
    cfg_dir = dest_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    protocol = """# Bajo Nervión PM10 TFM protocol
study_area: Bajo Nervión
zone: 2
contaminant: PM10
n_stations: 18
horizon: 24
train_ratio: 0.70
val_ratio: 0.10
test_ratio: 0.20
primary_metric: ma24_rmse
ranking: equal_station_mean

# Official look-back (ACF) + sensitivity grid
seq_len_official: 48
seq_len_grid: [24, 48, 72, 96]

feature_packs_official: [F1, F2, F3, F4, F5]
feature_packs_peak_extras: [F3I, F3IL, F3IL_pw, F5I]

models:
  - informer2020
  - gat_informer
  - airformer
  - xgboost

seeds:
  default: 2024
"""
    (cfg_dir / "bajo_nervion_pm10.yaml").write_text(protocol, encoding="utf-8")
    # backward-compatible alias used by some scripts
    (cfg_dir / "zone2_pm10.yaml").write_text(protocol, encoding="utf-8")
    for L in (24, 48, 72, 96):
        (cfg_dir / f"seq_len_{L}.yaml").write_text(
            f"# Override look-back for sensitivity run\n"
            f"study_area: Bajo Nervión\n"
            f"seq_len: {L}\n"
            f"horizon: 24\n"
            f"zone: 2\n"
            f"contaminant: PM10\n",
            encoding="utf-8",
        )
    (cfg_dir / "horizon_48.yaml").write_text(
        """# Supplementary forecast horizon (H=48 h)
study_area: Bajo Nervión
horizon: 48
seq_len: 48
zone: 2
contaminant: PM10
metrics: [ma24_rmse, ma24_d1, ma24_d2]
""",
        encoding="utf-8",
    )


def write_gitignore(dest_root: Path) -> None:
    (dest_root / ".gitignore").write_text(
        "\n".join(
            [
                "venv/",
                "__pycache__/",
                "*.pyc",
                ".ipynb_checkpoints/",
                ".env",
                "data/training/zone_2/**/seq_len_*/",
                "data/training/zone_2/airformer/",
                "data/training/zone_2/informer2020/",
                "data/training/zone_2/gat_informer*/",
                "data/training/zone_2/dataset_zone_2_PM10.csv",
                "results/**/checkpoints/",
                "results/**/*.pth",
                "results/**/*.pt",
                "results/**/*.npy",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_data_readme(dest_root: Path) -> None:
    (dest_root / "data" / "README.md").write_text(
        """# Data (public)

Air-quality and meteorology observations for Basque Country monitoring
stations are public (Euskadi open data / OpenData Euskadi). They may be
redistributed for research reproduction.

## What is shipped

**Definitive TFM table** for the Bajo Nervión study area (recommended entry point):

- `training/zone_2/dataset_zone_2_PM10.csv` (~146 MB uncompressed)
- `training/zone_2/dataset_zone_2_PM10.csv.zip` (~20 MB) — what git should track
- `training/zone_2/prepare_training_manifest_PM10.json` — station list / prep metadata

Unzip before training:

```bash
cd data/training/zone_2
python -c "import zipfile; zipfile.ZipFile('dataset_zone_2_PM10.csv.zip').extractall('.')"
```

## What is not shipped (and why)

| Asset | Reason |
|-------|--------|
| Full `data/raw/` dump (~500+ MB) | Too large for a light GitHub clone; identical public sources |
| Model tensors under `data/training/zone_2/<model>/` | Tens of GB; rebuild from the definitive CSV |

To rebuild tensors from the definitive CSV, use the export scripts under
`src/data/` (Informer / AirFormer / GAT-Informer / XGBoost feature packs).

Dartboard assignment for AirFormer DS-MSA:

- `graphs/zone2_18_dartboard/assignment.npy`
""",
        encoding="utf-8",
    )


def write_readme(dest_root: Path) -> None:
    (dest_root / "README.md").write_text(
        f"""# {REPO_NAME}

PM₁₀ forecasting in **Bajo Nervión** — MSc Final Project reproduction package (Alejandro Padro).

Minimal package to reproduce the thesis experiments:

- Informer, GAT-Informer, AirFormer, XGBoost
- Feature packs F1–F5 (+ peak extras F3IL / F5I)
- Look-back sensitivity \\(L \\in \\{{24,48,72,96\\}}\\) (official \\(L=48\\) from ACF)
- Supplementary horizon H=48 and AirFormer MSE loss sensitivity
- Lead-time error and MA24 ≥45 µg/m³ exceedance diagnostics
- Equal station-mean MA24 ranking, skill vs persistence, error by PM₁₀ amplitude

This is a **reviewer-facing** extract of the full research workspace (not the
entire lab dump). See **`REVIEWER.md`** for a one-page evaluator guide.

**Repository:** {GITHUB_URL}

## Quick start

```bash
python -m venv venv
# Windows: .\\venv\\Scripts\\Activate.ps1
source venv/bin/activate
pip install -r requirements.txt

cd data/training/zone_2
python -c "import zipfile; zipfile.ZipFile('dataset_zone_2_PM10.csv.zip').extractall('.')"
cd ../../..
```

## What is included

| Path | Role |
|------|------|
| `data/training/zone_2/dataset_zone_2_PM10.csv.zip` | Definitive public dataset (~20 MB) |
| `analysis/eda` | EDA scripts / notebooks |
| `analysis/feature_selection/zone_2/PM10` | ACF, lag correlations, F1–F5 readiness |
| `analysis/acf/zone_2/PM10` | Shared ACF / `seq_len` recommendation |
| `configs/bajo_nervion_pm10.yaml` | Official protocol |
| `configs/seq_len_{{24,48,72,96}}.yaml` | Look-back sensitivity |
| `configs/horizon_48.yaml` | Supplementary H=48 protocol |
| `src/` | Training, models, comparison metrics |
| `analysis/comparison/` | Paper figures |
| `results/comparison/zone_2/**/tables` | Metric tables (incl. lead-time & exceedance) |
| `docs/figures_print/` | Final PNGs for the manuscript |
| `docs/tfm_manuscript_overleaf_v02.tex` | **Latest** manuscript source (v02) |
| `results/comparison/zone_2/horizon_48/` | H=24 vs H=48 comparison tables |
| `results/airformer/zone_2/ZONE2_PM10/F*_mse/` | MSE loss sensitivity (AirFormer) |

**Not included:** full raw dump (~500 MB) — public and rebuildable; see `data/README.md`.
Model tensor caches (tens of GB) are also omitted.

## Reproduction levels

1. **Tables & figures (CPU, minutes)** — regenerate plots from shipped CSVs:
   ```bash
   python analysis/comparison/plot_skill_with_peak_extras.py
   python analysis/comparison/error_by_pm10_amplitude_packs.py
   python scripts/analysis/plot_headline_lead_rmse_hit.py
   python src/experiments/compare_horizon_24_48.py
   ```
2. **EDA / ACF** — run `analysis/eda/eda_zone_contaminant.py` (PM10, Bajo Nervión)
   and inspect `analysis/feature_selection/zone_2/PM10/`.
3. **Full retrain (GPU, hours–days)** — prepare model tensors from the definitive
   CSV, then train with `seq_len` overrides from `configs/seq_len_*.yaml`.
   See model READMEs under `models/*/README.md` and `scripts/hpc/`.

## Protocol (summary)

- Bajo Nervión, 18 PM₁₀ stations; official horizon H=24 h (+ supplementary H=48)
- Official look-back L=48 h (ACF); sensitivity L∈{{24,48,72,96}}
- AirFormer trained with official MAE; MSE variant reported as sensitivity check
- Chronological 70/10/20 split
- Primary metric: equal-station-mean MA24 RMSE

## Upstream code

- AirFormer: https://github.com/yoshall/AirFormer
- GAT-Informer: https://github.com/ChengqingYu/GAT-Informer
- Informer: Zhou et al. (AAAI 2021)

## License / data

Observational data are from public Basque air-quality / meteorology sources.
Please cite the thesis and the upstream model papers when reusing this package.
""",
        encoding="utf-8",
    )


def write_reviewer(dest_root: Path) -> None:
    (dest_root / "REVIEWER.md").write_text(
        f"""# Notes for TFM evaluators

**Package:** `{REPO_NAME}` — reviewer snapshot (manuscript v02, September 2026)

## Start here

1. **Manuscript (latest):** `docs/tfm_manuscript_overleaf_v02.tex`
2. **Figures in the PDF:** `docs/figures_print/` (incl. `fig_lead_rmse_hit_headline.png`)
3. **Tables backing Results:** `results/comparison/zone_2/all_models/tables/`
4. **Protocol:** `configs/bajo_nervion_pm10.yaml`

## Key supplementary analyses

| Topic | Where |
|-------|-------|
| Manuscript v02 (recortes + MSE + Saharan gap + lead-time) | `docs/tfm_manuscript_overleaf_v02.tex` |
| Loss-function check (AirFormer MAE vs MSE) | `results/airformer/zone_2/ZONE2_PM10/F*_mse/` |
| Supplementary horizon H=48 | `results/comparison/zone_2/horizon_48/`, `configs/horizon_48.yaml` |
| Look-back sensitivity L∈{{24,48,72,96}} | `results/comparison/zone_2/seq_len_sensitivity/` |
| Lead-time RMSE / hourly POD (≥45 µg/m³) | `tables/lead_*`, `scripts/analysis/plot_headline_lead_rmse_hit.py` |
| MA24 ≥45 µg/m³ exceedance (per station) | `tables/exceedance_ma24_thr45_*` |

## Reproduce without GPU (≈10 min)

```bash
python analysis/comparison/plot_skill_with_peak_extras.py
python analysis/comparison/error_by_pm10_amplitude_packs.py
python scripts/analysis/plot_headline_lead_rmse_hit.py
python src/experiments/compare_horizon_24_48.py
```

## Data

Unzip `data/training/zone_2/dataset_zone_2_PM10.csv.zip` before any training.
Full tensor caches and checkpoints are **not** shipped (rebuild from CSV).

**Repository:** {GITHUB_URL}
""",
        encoding="utf-8",
    )


def patch_manuscript_github_url(dest_root: Path) -> None:
    """Point manuscript availability sentence to this repository."""
    tex = dest_root / "docs" / "tfm_manuscript_overleaf_v02.tex"
    if not tex.exists():
        return
    text = tex.read_text(encoding="utf-8")
    old_urls = [
        "https://github.com/apadroh/tfm-pm10-bajo-nervion-repro",
        "https://github.com/apadroh/2026_FPD_Ales_Padro_Bajo_Nervion",
    ]
    for old in old_urls:
        text = text.replace(old, GITHUB_URL)
    text = text.replace(
        "definitive zone-2 PM$_{10}$ dataset",
        "definitive Bajo Nervi\\'on PM$_{10}$ dataset",
    )
    text = text.replace(
        "definitive zone-2 PM$_{10}$ dataset",
        "definitive Bajo Nervi\\'on PM$_{10}$ dataset",
    )
    tex.write_text(text, encoding="utf-8")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument("--clean", action="store_true", help="Remove dest before export")
    args = p.parse_args()
    dest: Path = args.dest.resolve()
    if args.clean and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Exporting reproduction package -> {dest}")
    for rel_src, rel_dst in COPY_PATHS:
        src = ROOT / rel_src
        if not src.exists():
            print(f"  SKIP missing {rel_src}")
            continue
        print(f"  copy {rel_src} -> {rel_dst}")
        copy_tree(src, dest / rel_dst)

    zpath = zip_definitive_csv(dest)
    print(f"  definitive CSV zip -> {zpath.relative_to(dest)}")
    copy_xgboost_summaries(dest)
    copy_airformer_mse_summaries(dest)
    write_configs(dest)
    write_gitignore(dest)
    write_data_readme(dest)
    write_readme(dest)
    write_reviewer(dest)
    patch_manuscript_github_url(dest)

    shutil.copy2(Path(__file__), dest / "scripts" / "export_repro_package.py")

    for d in ["src", "src/models", "scripts"]:
        (dest / d).mkdir(parents=True, exist_ok=True)
        init = dest / d / "__init__.py"
        if d != "scripts" and not init.exists():
            init.write_text("", encoding="utf-8")

    total = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"Done. Package size ~ {total/1e6:.1f} MB at {dest}")
    print(f"GitHub URL: {GITHUB_URL}")


if __name__ == "__main__":
    main()
