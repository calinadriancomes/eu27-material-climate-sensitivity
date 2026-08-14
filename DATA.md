# Data

The analysis covers the 27 current EU Member States annually from 2010 through 2023. The fixed source snapshot was acquired from Eurostat on **8 August 2026** and is stored unchanged as compressed JSON-stat files in `data/raw/`.

The analysis uses:
- circular material use rate (CMUR);
- domestic material consumption (DMC);
- raw material consumption / material footprint (RMC);
- territorial greenhouse-gas emissions;
- consumption-based greenhouse-gas footprint;
- average population for the common per-capita denominator.

A production/residence greenhouse-gas series is retained for a diagnostic accounting bridge. Published per-capita series are retained as comparators.

`data/sources.csv` records, for each source entry, the Eurostat datacode, API query URL, resolved selectors, coverage, raw filename, and raw SHA-256. `data/data_dictionary.csv` documents the analysis fields. `data/panel_2010_2023.csv` is the reference analytic panel used to verify reconstruction from the fixed source responses.

For RMC, `data/provenance/` stores a source-controlled official-evidence register and a conservative country-level source/method registry. These records retain exact evidence locators and explicitly identify unresolved year-level boundaries; runtime reproduction does not infer method from observation-status flags or require live web access.

The fixed snapshot is included because Eurostat data may be revised and the live dissemination database serves the latest dataset version rather than a historical version of the database state used for this analysis. Author transformations from the stored responses to the analytic panel are implemented in `code/01_prepare_data.py` and checked against the reference panel.

## Reuse and attribution

Eurostat permits reuse of statistical data and metadata for commercial and non-commercial purposes when the source is acknowledged, subject to the conditions and exceptions in its copyright and reuse notice. When data are modified, the modification should be made clear. See:

- Eurostat copyright and reuse notice: `https://ec.europa.eu/eurostat/help/copyright-notice`
- Eurostat API introduction: `https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-introduction`

The repository's MIT License applies only to author-created software. It does **not** relicense Eurostat statistical data or metadata. No dataset DOI is asserted here; the exact datacodes and source URLs used in this snapshot are recorded in `data/sources.csv`.
