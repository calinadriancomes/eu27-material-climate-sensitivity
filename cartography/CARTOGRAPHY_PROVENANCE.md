# Cartography provenance

Publication-facing maps use **Natural Earth Admin 0 Countries, 1:10m, version 5.1.1**.

Frozen source archive:

- file: `cartography/source/ne_10m_admin_0_countries.zip`
- SHA-256: `ce1ac7036499a0edd641fbc093cd209a98f96a49d2eca8480aaacad35138a7f6`
- source CRS: WGS 84 / EPSG:4326
- public source page: `https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/`
- Natural Earth terms: public-domain map data; see `https://www.naturalearthdata.com/about/terms-of-use/`

## EU27 join

`COUNTRY_CODE_MAP.csv` contains the explicit scientific-key to Natural-Earth mapping. The 27/27 join check is recorded in `../MAP_JOIN_PARITY.csv` and requires:

- expected EU27: 27;
- matched: 27;
- missing: 0;
- duplicate country matches: 0;
- one unique geometry assignment per Member State;
- no scientific-value recomputation.

Eurostat uses `EL` for Greece. Natural Earth uses ISO `GR`; this translation is explicit. France is matched with `ADM0_A3=FRA` and `ADMIN=France` so the sovereign-country record is selected rather than other French dependency records.

## Rendering transformation

For the publication maps only, the Natural Earth polygons are:

1. clipped to the European map frame (longitude -12.8 to 35.8, latitude 34.0 to 72.5) so remote non-European territories do not determine the plotting extent;
2. projected to ETRS89 / LAEA Europe (EPSG:3035);
3. simplified at 2.5 km tolerance with topology preservation to keep vector files compact at journal scale.

Cyprus and Malta remain in their true geographic locations. No inset or synthetic country relocation is used. The geometric transformation affects map support only; quantitative values are joined by the fixed scientific country key after the 27/27 join check.

For the retained map figures, the manuscript/Supplement captions should include the following jurisdiction note:

> Map lines delineate study areas and do not necessarily depict accepted national boundaries.
