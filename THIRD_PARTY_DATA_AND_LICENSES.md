# Third-party data and license scope

## Author-created software

The root `LICENSE` applies to author-created software and documentation in this repository. It does not relicense third-party statistical or cartographic data.

## Eurostat statistical data and metadata

The frozen statistical source responses in `data/raw/` were obtained from Eurostat. Eurostat's copyright and reuse notice authorizes reuse subject to its stated conditions and exceptions, including source acknowledgement and disclosure of modifications where applicable.

Source: https://ec.europa.eu/eurostat/help/copyright-notice

This repository identifies Eurostat as the source and records the exact dataset codes, selectors, source URLs, acquisition snapshot, and source-file checksums in `data/sources.csv` and `DATA.md`. The transformations from source responses to the analytic panel are author-created and are documented in code.

## Natural Earth cartography

The publication map layer uses Natural Earth Admin 0 Countries, 1:10m, version 5.1.1. Natural Earth states that its raster and vector map data are in the public domain and free for use in any type of project.

Sources:

- https://www.naturalearthdata.com/about/terms-of-use/
- https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/

Frozen archive in this repository:

`cartography/source/ne_10m_admin_0_countries.zip`

The archive's SHA-256 is recorded in `cartography/CARTOGRAPHY_PROVENANCE.md` and the repository checksum manifest.

## No implied relicensing

Redistribution of a source snapshot inside this research repository is not a claim of ownership over the source data. Users should consult the source-provider terms when reusing third-party data independently of the accompanying analysis code.
