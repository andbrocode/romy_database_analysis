# ROMY Database Analysis

Notebooks and scripts for rotational seismology analysis using the [ROMY](https://www.lmu.de/romy) ring laser and co-located G-ring (GR.WET / BW.RLAS) instrumentation. The workflow covers event configuration, waveform download, batch processing (backazimuth, correlation, tilt correction, dispersion, spectra), ROMY–ADR comparison, body-wave polarization, ROMY–G-ring cross-correlation (CCF), and the summary figures for the 19-event dataset.

All paths below are relative to this repository root (`romy_database_analysis/`).

The 674-event G-ring backazimuth catalog is not part of this repository. G-ring miniSEED is included for 14 of the 19 events (missing: 2023-12-02, 2025-12-06, 2025-12-12, 2026-01-07, 2026-04-20). Array data are absent for 2019-05-14.


## Requirements

- **Python** 3.10 (recommended; see [`environment.yml`](environment.yml))
- **`sixdegrees`** — stream handling, array-derived rotation (ADR), and analysis utilities (used by all batch scripts and most notebooks)
- **Conda** (recommended) or a virtualenv with pip
- Network access for FDSN waveform download (LMU, BGR, etc.); some data sources require institutional access or credentials
- **Jupyter** for interactive notebooks


## Environment setup

### Option A — Conda environment (recommended)

The file [`environment.yml`](environment.yml) is a full conda export of the project environment (name: `sixdegrees`). It includes Python 3.10, ObsPy, Jupyter/JupyterLab, Matplotlib, SciPy, Cartopy, `disba`, and related dependencies.

The analysis was run with **`sixdegrees` 1.0.3** ([GitHub](https://github.com/andbrocode/SixDegreesOfFreedom), [Zenodo](https://doi.org/10.5281/zenodo.20334505)). The export still lists `sixdegrees==0.1.1` under `pip:`. After creating the environment, install version 1.0.3 over that pin.

**Create and activate:**

```bash
cd romy_database_analysis
conda env create -f environment.yml
conda activate sixdegrees
pip install -e /path/to/SixDegreesOfFreedom   # version 1.0.3
```

**Update an existing environment:**

```bash
conda activate sixdegrees
conda env update -f environment.yml --prune
pip install -e /path/to/SixDegreesOfFreedom   # version 1.0.3
```

**Verify:**

```bash
python -c "import sixdegrees; print(sixdegrees.__version__)"
jupyter lab   # or: jupyter notebook
```

**Regenerate** `environment.yml` after adding or upgrading packages:

```bash
conda activate sixdegrees
conda env export > environment.yml
```

For a smaller, more portable export (explicit installs only, fewer build pins):

```bash
conda env export --from-history > environment.yml
```

**Note:** The export is platform-specific (Linux, conda-forge). On macOS or Windows you may need to adjust channels or recreate the env from a shorter spec and install `sixdegrees` manually (Option B or C).

### Option B — Install `sixdegrees` with pip (no conda)

Use this if you already have a Python ≥ 3.9 environment with ObsPy, NumPy, SciPy, Matplotlib, PyYAML, and Cartopy (or install them separately):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install obspy jupyter matplotlib scipy cartopy pyyaml pandas disba
pip install -e /path/to/SixDegreesOfFreedom   # version 1.0.3
```

`sixdegrees` 1.0.3 is the version used for the figures in this repository.

### Option C — Editable install from source

If you develop `sixdegrees` locally or clone it from your institution's repository:

```bash
conda activate sixdegrees   # or your virtualenv
git clone https://github.com/andbrocode/SixDegreesOfFreedom.git
cd sixdegrees
pip install -e ".[dev,cartopy]"
```

Then return to this project directory and run scripts/notebooks as usual.

### Array configuration

Station geometry, channels, and default FDSN clients are defined in [`config_array_romy.yml`](config_array_romy.yml).


## Quick Start

Typical batch pipeline for events listed in [`events_list.yml`](events_list.yml). Run from the repository root with the environment activated (`conda activate sixdegrees`):

```bash
# 1. Per-event configs (interactive, once per new event)
jupyter notebook 01_XROMY_make_config.ipynb

# 2. Download waveforms
jupyter notebook 02_XROMY_get_data.ipynb    # ROMY data (internal LMU access only)
jupyter notebook 02_XADR_get_data.ipynb     # ADR array data + per-event ADR
python 02_XGRING_get_data.py                # G-ring data from FDSN (online)

# 3. Batch analysis (adjust parameters in config files or inside scripts)
python 04_XROMY_event_analysis_baz.py
python 04_XROMY_event_analysis_cc.py          # refreshes results_correlation.csv (tilt variance reduction)
python 04_XROMY_event_analysis_vel.py
python 04_XGRING_event_analysis_vel.py
python 04_XROMY_event_analysis_spec.py
python 04_XROMY_romy_adr_comparison.py

# Override the default event list without editing YAML:
# python 04_XROMY_event_analysis_baz.py --events 20240314,20240322
# python 04_XROMY_event_analysis_cc.py --events-file my_events.yml
# python 04_XROMY_event_analysis_vel.py --events-file my_events.yml   # vel scripts: --events-file only
# 04_XROMY_romy_adr_comparison.py always reads events_list.yml (no CLI override)

# 4. Summary figures (Jupyter)
cp output/results_dispersion.csv output/results_dispersion_romy.csv   # after a fresh velocity run
jupyter notebook 05_XROMY_plot_correlation_results.ipynb
jupyter notebook 05_XROMY_plot_tilt_correction.ipynb
jupyter notebook 05_XROMY_plot_dispersion_results.ipynb
jupyter notebook 05_XROMY_plot_waveforms_zoomin.ipynb
jupyter notebook 05_XROMY_plot_ccf_romy_gring.ipynb   # 20240422 and 20240815
```

**Correlation table:** both `04_XROMY_event_analysis_baz.py` and `04_XROMY_event_analysis_cc.py` write `output/results_correlation.csv`. The tilt-correction figure reads the variance-reduction columns written by the CC script, so run that script after the backazimuth script.

**Dispersion CSV naming:** `04_XROMY_event_analysis_vel.py` writes `output/results_dispersion.csv`. `05_XROMY_plot_dispersion_results.ipynb` reads `output/results_dispersion_romy.csv` (the file already in `output/`). Copy or rename after a fresh velocity run. The publication curves use the `velocity_median` column.


## Workflow overview

Files are numbered by processing stage and prefixed by dataset (`XROMY`, `XGRING`, `XADR`):

| Stage | Purpose |
| ----- | ------- |
| `01_` | Build per-event YAML configuration files |
| `02_` | Download / prepare waveform data per event |
| `03_` | Event-by-event interactive analysis (notebooks) |
| `04_` | Batch event analysis and ROMY–ADR comparison |
| `05_` | Aggregate plotting and result visualization |
| `06_` | Reporting / summary tables |


## Python scripts

Run as `python <script>.py` from the repository root.

Most scripts read `events_list.yml` by default. Override options (see each script's `--help`):

| Script | `--events-file` | `--events` |
| ------ | :-------------: | :--------: |
| `02_XGRING_get_data.py` | yes | yes |
| `04_XROMY_event_analysis_baz.py` | yes | yes |
| `04_XROMY_event_analysis_cc.py` | yes | yes |
| `04_XROMY_event_analysis_vel.py` | yes | — |
| `04_XGRING_event_analysis_vel.py` | yes | — |
| `04_XROMY_event_analysis_spec.py` | yes | yes |
| `04_XROMY_romy_adr_comparison.py` | — | — |

| Script | Input | Output |
| ------ | ----- | ------ |
| `02_XGRING_get_data.py` | `events_list.yml`, per-event `config_romy_*.yml` (timing) | `events/event_*/gring_YYYYMMDD.mseed` |
| `04_XROMY_event_analysis_baz.py` | `config_romy_*.yml`, ROMY MSEED | `output/results_backazimuth.csv`, `output/results_correlation.csv`, `output/restuls_correlation_by_window.csv` |
| `04_XROMY_event_analysis_cc.py` | `config_romy_*.yml`, ROMY MSEED | `output/results_correlation.csv` (CC and tilt variance reduction), per-event tilt figures |
| `04_XROMY_event_analysis_vel.py` | `config_romy_*.yml`, ROMY MSEED | `output/results_dispersion.csv` |
| `04_XGRING_event_analysis_vel.py` | `config_gring_*.yml`, G-ring MSEED | `output/results_dispersion_gring.csv` |
| `04_XROMY_event_analysis_spec.py` | `config_romy_*.yml`, ROMY MSEED | `events/event_*/spectral_analysis_YYYYMMDD.json` |
| `04_XROMY_romy_adr_comparison.py` | ROMY + ADR configs and waveforms, `events_list.yml` | `output/results_romy_adr_comparison.csv`, `events/event_*/figures/*_romy_adr_comparison.png` |

G-ring download window: GR.WET translation and BW.RLAS rotation, −5 min to +150 min relative to origin time.


## Event configuration

Each `events/event_YYYYMMDD/config_romy_YYYYMMDD.yml` typically specifies:

- Origin time, magnitude, analysis window (`tbeg` / `tend`)
- MSEED path (`path_to_mseed_file`), figure output directory
- Component SEED IDs (`rot_seed`, `tra_seed`), processing bands (`fmin` / `fmax`)
- Backazimuth / CC parameters (`baz_step`, `cc_threshold`, `twin_sec`, `twin_overlap`)

G-ring and ADR configs mirror event timing with instrument-specific station lists. Regenerate configs with `01_XROMY_make_config.ipynb` when adding events to `events_list.yml`.


## Notebooks

### `01_*` Configuration

- **`01_XROMY_make_config.ipynb`** — Creates/updates `events/event_*/config_romy_*.yml`, `config_gring_*.yml`, and related metadata from the master event list.

### `02_*` Data acquisition

- **`02_XROMY_get_data.ipynb`** — Fetches ROMY rotational/translation MSEED into each event folder.
- **`02_XADR_get_data.ipynb`** — Downloads ADR array waveforms using per-event `config_adr_*.yml` and `config_array_romy.yml`.

### `03_*` Per-event interactive analysis

- **`03_XROMY_event_analysis.ipynb`** — Waveforms, spectra, backazimuth, maps, dispersion for one ROMY event.
- **`03_XGRING_event_analysis.ipynb`** — G-ring waveforms, CC, backazimuth, Love-wave velocity / dispersion.
- **`03_modify_dispersion_models.ipynb`** — Builds `data/model_prem.csv`, `data/model_local.csv`, `data/dispersion_prem.csv`, `data/dispersion_local.csv` from crustal model files.

### `04_*` Comparison (interactive)

- **`04_XROMY_romy_adr_comparison.ipynb`** — Interactive ROMY vs ADR; writes `output/romy_adr_comparison_results.csv` and per-event figures (batch script uses `output/results_romy_adr_comparison.csv`).

### `05_*` Plotting / result visualization

| Notebook | Primary input | Typical output |
| -------- | --------------- | -------------- |
| `05_XROMY_plot_dispersion_results.ipynb` | `output/results_dispersion_romy.csv`, `data/dispersion_*.csv` | `output/figures/dispersion/`; publication curves in `publication/` |
| `05_XGRING_plot_dispersion_results.ipynb` | `output/results_dispersion_gring.csv` | `output/figures/dispersion/`; Love mid/max panel in `publication/` |
| `05_XROMY_plot_backazimuth_results.ipynb` | `output/results_backazimuth.csv` | `output/figures/backazimuth/`; summary panel in `publication/` |
| `05_XROMY_plot_correlation_results.ipynb` | `output/results_correlation.csv` | `output/figures/correlation/`; combined panels in `publication/` |
| `05_XROMY_plot_tilt_correction.ipynb` | `output/results_correlation.csv` | `output/figures/tilt_correction/` |
| `05_XROMY_plot_spectra_results.ipynb` | `events/event_*/spectral_analysis_*.json`, `data/low_noise_models.csv` | `output/figures/spectra/`; ASD panel in `publication/` |
| `05_XROMY_plot_waveforms_overview.ipynb` | `events_list.yml`, ROMY configs / MSEED | `output/figures/waveforms/` |
| `05_XROMY_plot_waveforms_zoomin.ipynb` | Per-event ROMY MSEED | `output/figures/event/` (P/S zoom, SH panel, polarization) |
| `05_XROMY_plot_maps.ipynb` | Station geometry, `data/ROMY_30m_DEM.tif` | `output/figures/publication/romy_array_maps.png` |
| `05_XROMY_plot_ccf_romy_gring.ipynb` | ROMY/G-ring configs and MSEED | `output/figures/ccf/`; v2 comparison in `publication/` |
| `05_XROMY_plot_adr_comparison_results.ipynb` | `output/results_romy_adr_comparison.csv` | `output/figures/adr/`; three-metric panel in `publication/` |

### `06_*` Reporting

- **`06_XROMY_make_event_config_table.ipynb`** — LaTeX-ready event/config table (`output/tables/events_config_table.tex`).


## Directory layout

```
romy_database_analysis/
├── environment.yml
├── events_list.yml
├── config_array_romy.yml
├── 01_XROMY_make_config.ipynb
├── 02_XROMY_get_data.ipynb
├── 02_XADR_get_data.ipynb
├── 02_XGRING_get_data.py
├── 03_XROMY_event_analysis.ipynb
├── 03_XGRING_event_analysis.ipynb
├── 03_modify_dispersion_models.ipynb
├── 04_*.py / 04_*.ipynb
├── 05_*.ipynb
├── 06_XROMY_make_event_config_table.ipynb
│
├── events/
│   └── event_YYYYMMDD/
│       ├── config_romy_*.yml
│       ├── config_gring_*.yml
│       ├── config_adr_*.yml
│       ├── romy_data_*.yml
│       ├── romy_*.mseed
│       ├── gring_*.mseed
│       ├── spectral_analysis_*.json
│       ├── ellipticity_analysis_*.json
│       └── figures/
│
├── output/
│   ├── results_*.csv
│   ├── figures/
│   │   ├── publication/          # manuscript panels
│   │   ├── adr/
│   │   ├── backazimuth/
│   │   ├── ccf/
│   │   ├── correlation/
│   │   ├── dispersion/
│   │   ├── event/                # body-wave zooms and polarization
│   │   ├── spectra/
│   │   ├── tilt_correction/
│   │   ├── waveforms/
│   │   └── misc/                 # sketches and other leftovers
│   └── old/
│
├── data/
│   ├── model_*.csv
│   ├── dispersion_*.csv
│   ├── low_noise_models.csv
│   ├── *.crh                     # local crustal models
│   └── ROMY_30m_DEM.tif          # site map background (05_XROMY_plot_maps.ipynb)
│
└── metadata/
    └── station_*.xml
```


## Note on AI assistance

Parts of this project were developed with AI assistance. All AI-generated suggestions were reviewed, supervised, and validated by the project author before use.
