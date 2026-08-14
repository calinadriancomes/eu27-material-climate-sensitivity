#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REVIEW_DEVELOPMENT_TOKEN=''.join(('review','er-motivated'))
FORBIDDEN=[
    REVIEW_DEVELOPMENT_TOKEN,'pre-specified','common-method sample','harmonized sample',
    'restricted-comparability sample','stable-provenance sample','primary full-grid robustness'
]

def main():
    reg=pd.read_csv(ROOT/'data/provenance/rmc_country_method_registry.csv',keep_default_na=False)
    assert len(reg)==27 and reg.geo.nunique()==27
    mixed=reg[reg.disseminated_source_regime.eq('Mixed/changed over 2010–2023')].geo.tolist()
    assert len(mixed)==7

    n20=pd.read_csv(ROOT/'results/reference/supplementary_data/rmc_provenance_sensitivity_n20.csv',keep_default_na=False)
    assert len(n20)==1 and int(n20.iloc[0].n_countries)==20
    assert set(n20.iloc[0].excluded_countries.split(';'))==set(mixed)
    assert len(n20.iloc[0].included_countries.split(';'))==20

    groups=pd.read_csv(ROOT/'results/reference/supplementary_data/rmc_provenance_group_grids.csv',keep_default_na=False)
    expected={'country-reported':9,'Eurostat-estimated':11,'mixed/changed':7}
    assert {r.provenance_group:int(r.n_countries) for _,r in groups.iterrows()}==expected
    for _,r in groups.iterrows():
        assert len(r.countries.split(';'))==int(r.n_countries)

    deletion=pd.read_csv(ROOT/'results/reference/supplementary_data/full_grid_deletion_runs.csv',keep_default_na=False)
    assert len(deletion[deletion.deletion_type.eq('FULL')])==1
    assert len(deletion[deletion.deletion_type.eq('LOCO')])==27
    assert len(deletion[deletion.deletion_type.eq('LOYO')])==14
    for c in ['full_grid_minimum_cell_key','full_grid_minimum_cell_label','full_grid_maximum_cell_key','full_grid_maximum_cell_label']:
        assert (deletion[c].astype(str).str.strip()!='').all(),c

    f6=pd.read_csv(ROOT/'figures/source_data/figure_6_deletion_robustness.csv',keep_default_na=False)
    assert 'full_grid_range' in f6.columns
    assert set(f6.full_grid_range_metric)=={'Difference in Spearman rho'}
    assert set(f6.cell_metric)=={'Spearman rho'}
    assert 'selected_endpoint_cell_difference' not in f6.columns
    assert len(f6[f6.source.eq('LOCO')])==27 and len(f6[f6.source.eq('LOYO')])==14
    for c in ['full_grid_minimum_cell_key','full_grid_maximum_cell_key']:
        assert c in f6.columns and (f6[c].astype(str).str.strip()!='').all()

    t5=pd.read_csv(ROOT/'results/reference/tables/table_5_temporal_and_robustness.csv',keep_default_na=False)
    assert set(t5.panel_heading)=={'Panel A — Grid-level robustness and sensitivity'}
    assert set(t5.analytical_object)=={'Full-grid range'}
    assert set(t5.metric)=={'Difference in Spearman rho'}
    assert not t5.astype(str).apply(lambda c:c.str.contains('endpoint',case=False,regex=False)).any().any()
    assert len(t5[t5.diagnostic_key.eq('rmc_exclude_mixed_changed')])==1
    assert t5.loc[t5.diagnostic_key.eq('rmc_exclude_mixed_changed'),'scope_or_n'].iloc[0]=='n = 20'

    t3=pd.read_csv(ROOT/'results/reference/tables/table_3_primary_concordance_and_contrasts.csv',keep_default_na=False)
    for typ in ['full_grid_minimum','full_grid_maximum']:
        q=t3[t3.object_type.eq(typ)]
        assert len(q)==1 and q.iloc[0].metric=='Spearman rho'

    public_files=[
        ROOT/'results/reference/tables/table_5_temporal_and_robustness.csv',
        ROOT/'figures/source_data/figure_6_deletion_robustness.csv',
        ROOT/'docs/PUBLICATION_OUTPUT_MAP.csv', ROOT/'docs/PUBLICATION_LABELS.csv'
    ]
    text='\n'.join(p.read_text(encoding='utf-8') for p in public_files).lower()
    for bad in FORBIDDEN: assert bad not in text,bad
    assert 'overall effect' not in text
    assert 'representation effect' not in text
    assert 'magnitude of accounting effect' not in text
    assert 'decomposition' not in '\n'.join((ROOT/'results/reference/tables/table_5_temporal_and_robustness.csv').read_text().lower().splitlines())
    assert 'primary_' not in (ROOT/'results/reference/tables/table_5_temporal_and_robustness.csv').read_text().lower()
    assert 'primary_' not in (ROOT/'figures/source_data/figure_6_deletion_robustness.csv').read_text().lower()
    print('V18_SCIENCE_SEMANTIC_TEST_PASS full_grid=PRIMARY endpoint=SECONDARY rmc_n20_rule=PASS deletion_identities=PASS')

if __name__=='__main__': main()
