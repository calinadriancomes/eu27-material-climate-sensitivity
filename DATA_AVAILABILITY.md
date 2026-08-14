# Data availability

The underlying statistical series originate from Eurostat. The exact API-response snapshots used for the 2010-2023 analysis are included in `data/raw/`, together with the recorded source queries in `data/sources.csv` and a data dictionary in `data/data_dictionary.csv`.

The repository therefore supports reproduction against the same frozen source bytes even if the live Eurostat dissemination database is revised later. `code/01_prepare_data.py` reconstructs the analytic panel from those snapshots, and the tests compare that reconstruction with the checked-in reference panel.

Publication-source tables and figure source data are generated from the analysis outputs. The manuscript Supplementary Information and Data S1 are separate publication products; the repository independently reproduces the computational analysis and publication source files.

RMC source/method provenance used for publication support is frozen in `data/provenance/`, including official evidence locators and access dates. These provenance records are part of the reproducible snapshot and are not fetched from live websites during execution.
