# Model-assisted series: reproducibility and comparability note

Raw material consumption (RMC) and the consumption-based greenhouse-gas footprint are both model-assisted official statistics, but their documented production structures are not identical.

For RMC, official Eurostat metadata states that country values may be nationally reported or estimated by Eurostat. Eurostat estimates use the country RME tool, while reporting countries may use the Eurostat tool or a country-specific model. The same metadata warns that differences among country estimation models can hamper cross-country comparability and describes temporal comparability within a country as stronger than geographic comparability in this respect. The repository therefore preserves a conservative country-level source/method registry in `data/provenance/rmc_country_method_registry.csv`, linked to exact official evidence locations in `data/provenance/rmc_official_evidence_register.csv`. Where the official material does not resolve exact year-level method boundaries, the registry states that limitation rather than inferring it from observation-status flags.

An exact independent EU27 × 2010–2023 reconstruction of the disseminated Eurostat Python-tool route is not treated as reproducible from the public materials bundled here. The public country-RME materials do not establish a byte/route-equivalent reconstruction for every disseminated country-year value, so this repository retains the official disseminated RMC series rather than substituting an approximate common-method reconstruction.

For the consumption-based GHG footprint, Eurostat documents a common modelling method across countries and complete-series re-estimation with releases. Model uncertainty remains, but the additional country-method heterogeneity documented for RMC is not assumed to apply in the same form.

The official source URLs, evidence sections/fields and access date supporting these statements are stored in the source-controlled evidence register. Reproduction does not require fetching those pages.
