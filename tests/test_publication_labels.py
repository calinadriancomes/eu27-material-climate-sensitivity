#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
REQ={
    'CMUR','DMC','RMC','TERR','CFOOT','D1','D2','D3','D_TOTAL','FULL_GRID_RANGE',
    'POINT_SIGN_DISAGREEMENT','LOYO_BOTH_SIGN_STABLE','BOOTSTRAP_DIAGNOSTIC_RANGE',
    'FULL_2010_2023','EARLY_2010_2016','LATE_2017_2023','CMUR_even_year',
}
def main():
    d=pd.read_csv(ROOT/'docs/PUBLICATION_LABELS.csv',keep_default_na=False)
    assert d.internal_key.is_unique and REQ.issubset(set(d.internal_key))
    q=d.set_index('internal_key')
    assert q.loc['D_TOTAL','public_label']=='Selected endpoint cell difference'
    assert q.loc['FULL_GRID_RANGE','public_label']=='Full-grid range'
    assert q.loc['POINT_SIGN_DISAGREEMENT','public_label']=='Point-slope sign disagreement'
    assert q.loc['CMUR_even_year','public_label']=='Common even-year sensitivity'
    assert q.loc['CMUR_even_year','short_public_label']=='Common even-year'
    assert q.loc['FULL_2010_2023','public_label']=='Full (2010–2023)'
    assert q.loc['EARLY_2010_2016','public_label']=='Early (2010–2016)'
    assert q.loc['LATE_2017_2023','public_label']=='Late (2017–2023)'
    for k in ['D1','D2','D3']:
        assert 'Supplement' in q.loc[k,'allowed_context']

    # Selected endpoint is secondary descriptive evidence: never label its reader-facing reference as primary.
    f5=pd.read_csv(ROOT/'figures/source_data/figure_s5_broader_robustness_summary.csv',keep_default_na=False)
    assert 'primary_FULL' not in set(f5.diagnostic)
    assert 'full_reference' in set(f5.diagnostic)
    svg=(ROOT/'figures/reference/figure_s5_broader_robustness_summary.svg').read_text(encoding='utf-8',errors='ignore')
    assert 'Primary 2010' not in svg and 'primary_FULL' not in svg
    assert 'Full 2010–2023 reference' in svg

    outmap=pd.read_csv(ROOT/'docs/PUBLICATION_OUTPUT_MAP.csv',keep_default_na=False)
    c=outmap[outmap.repository_source.eq('results/reference/supplementary_data/trend_interval_counts.csv')]
    drow=outmap[outmap.repository_source.eq('results/reference/supplementary_data/trend_interval_diagnostics.csv')]
    assert len(c)==1 and c.iloc[0].manuscript_or_supplement=='SUPPLEMENT'
    assert len(drow)==1 and drow.iloc[0].manuscript_or_supplement=='DATA_ONLY'
    print(f'PUBLICATION_LABELS_TEST_PASS rows={len(d)} endpoint_hierarchy=PASS interval_mapping=PASS')
if __name__=='__main__': main()
