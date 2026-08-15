# EU27 material-climate progress

This repository reproduces a longitudinal comparison of national material and greenhouse-gas progress across the EU27 for 2010–2023. It includes the fixed Eurostat source responses used in the analysis, analysis code, reference results, source tables and the renderer that generates manuscript and Supplementary Information figures.

## Run in Google Colab

The first notebook rebuilds and packages the analysis, including the reported secondary publication diagnostics. The second uses that package to reproduce the publication tables and figures from the same Git commit.

- Prepare and package the analysis: [![Open notebook 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/calinadriancomes/eu27-material-climate-sensitivity/blob/main/notebooks/01_prepare_material_climate_analysis.ipynb)
- Reproduce the final publication layer: [![Open notebook 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/calinadriancomes/eu27-material-climate-sensitivity/blob/main/notebooks/02_reproduce_material_climate_results.ipynb)

## Scientific scope

The analysis compares country progress orderings across five official series that represent different indicator constructs and accounting choices:

- circular material use rate (CMUR), a circularity-share indicator;
- domestic material consumption (DMC), direct material throughput;
- raw material consumption / material footprint (RMC), a final-demand material footprint;
- territorial greenhouse-gas emissions;
- consumption-based greenhouse-gas footprint.

The common element is the comparative procedure: the current EU27 country set, analysis window, favorable-direction convention, country-specific trend procedure, ranking rule and rank-concordance calculation. CMUR versus DMC is an indicator/construct comparison, DMC versus RMC changes the material accounting boundary and territorial versus consumption-based GHG changes the climate accounting perspective.

The primary system-level output is the symmetric Full-period 3 × 2 Spearman grid. The repository also provides symmetric conditional differences, within-axis concordance, country rank displacement, point-slope sign disagreement with leave-one-year sign stability, temporal companions and clearly separated deletion, estimator, temporal-support, denominator and scale/estimand checks. A joint residual-block bootstrap is supplied as a secondary Supplement/Data diagnostic rather than as a MAIN-table inferential framework.

## Reproduce the analysis

Create an environment with Python 3.11+ and install the pinned dependencies:

```bash
python -m pip install -r requirements-lock.txt
```

Then run:

```bash
python code/checks.py verify
python code/01_prepare_data.py
python tests/test_ties.py
python code/02_analyze_progress.py
python tests/test_reproduction.py
python code/05_build_extended_publication_evidence.py
python tests/test_extended_publication_evidence.py
```

The reproduced analysis is written to `work/analysis/`. Baseline numerical outputs are checked against the frozen reference results; the extended-evidence test independently regenerates and checks the reported secondary outputs.

## Reproduce publication sources and figures

After the analysis and extended-evidence steps:

```bash
python code/04_render_publication_figures.py \
  --analysis work/analysis \
  --output-root work/publication \
  --natural-earth cartography/source/ne_10m_admin_0_countries.zip
```

Verify and package the reproduced publication layer:

```bash
python code/checks.py compare-publication --publication work/publication
python code/checks.py package-publication \
  --publication work/publication \
  --output eu27-material-climate-publication-results.zip
```

The checked-in publication sources are under:

- `results/reference/tables/` — manuscript Tables 2–5 sources;
- `supplementary_tables/` — Tables S1–S9 sources;
- `results/reference/supplementary_data/` — detailed machine-readable Supplement/Data evidence, including bootstrap summaries; publication-results ZIPs expose these files under `supplementary_data/`;
- `figures/source_data/` — source data for Figures 1–6 and S1–S7;
- `figures/reference/` — publication artwork in PDF/PNG/SVG formats.

Run the publication tests with:

```bash
python tests/test_publication_tables.py
python tests/test_publication_figure_source_data.py
python tests/test_publication_artwork.py
python tests/test_publication_labels.py
python tests/test_map_join_parity.py
python tests/test_publication_renderer_empty_output_root.py
python tests/test_notebook_publication_contract.py
```

`docs/PUBLICATION_OUTPUT_MAP.csv` maps publication objects to repository sources. `docs/PUBLICATION_LABELS.csv` provides the publication label dictionary. `docs/MODEL_ASSISTED_SERIES_METHOD_NOTE.md` explains the reproducibility/comparability boundary for RMC and the consumption-based GHG footprint.

## Cartography

Publication maps use Natural Earth Admin 0 Countries, 1:10m, version 5.1.1. The exact source archive is frozen under `cartography/source/`. Scientific country keys are joined to map geometry through `cartography/COUNTRY_CODE_MAP.csv`; `MAP_JOIN_PARITY.csv` records the 27/27 join check. Eurostat `EL` is explicitly mapped to Natural Earth `GR`, and France is matched to the sovereign-country record rather than dependency records.

Map geometry is presentation support only. Changing boundary geometry does not change progress scores, ranks, correlations, path arithmetic, point-slope sign results or country comparisons.

## Data and licensing

See `DATA.md`, `DATA_AVAILABILITY.md` and `THIRD_PARTY_DATA_AND_LICENSES.md`. The MIT License covers author-created software only; third-party/source data retain their own reuse terms. RMC method-provenance evidence is frozen under `data/provenance/` so runtime reproduction does not depend on live website content.

## Citation

Citation metadata are provided in `CITATION.cff`. The archived `v1.0.0` release will provide the persistent version locator.

## Repository structure

The repository separates source data, analysis code, canonical reference outputs and publication-facing artifacts.
See `docs/METHOD_AND_OUTPUT_MAP.md` for the relationship among source snapshots, analysis outputs, extended publication evidence, publication tables, figure source data, artwork and verification tests.
