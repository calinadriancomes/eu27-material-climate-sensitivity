#!/usr/bin/env python3
from pathlib import Path
import importlib.util, shutil, tempfile
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXPECTED_MAIN=[
"table_2_representation_contract_source.csv","table_3_primary_concordance_and_contrasts.csv","table_4_country_consequences.csv","table_5_temporal_and_robustness.csv"]
EXPECTED_SUPP=[f"table_s{i}_" for i in range(1,9)]

def load_renderer():
    p=ROOT/"code"/"04_render_publication_figures.py"; spec=importlib.util.spec_from_file_location("publication_renderer",p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def compare(a,b,tol=1e-12):
    x=pd.read_csv(a,keep_default_na=False); y=pd.read_csv(b,keep_default_na=False)
    assert list(x.columns)==list(y.columns) and x.shape==y.shape,(a,b,x.shape,y.shape)
    for c in x.columns:
        xn=pd.to_numeric(x[c],errors="coerce"); yn=pd.to_numeric(y[c],errors="coerce")
        if xn.notna().sum()==len(x) and yn.notna().sum()==len(y):
            assert np.max(np.abs(xn.to_numpy(float)-yn.to_numpy(float)))<=tol,c
        else:
            assert (x[c].astype(str).to_numpy()==y[c].astype(str).to_numpy()).all(),c

def semantic_assertions(root):
    t3=pd.read_csv(root/'results/reference/tables/table_3_primary_concordance_and_contrasts.csv',keep_default_na=False)
    for typ in ['full_grid_minimum','full_grid_maximum']:
        q=t3[t3.object_type.eq(typ)]
        assert len(q)==1 and q.iloc[0].metric=='Spearman rho',(typ,q[['metric']].to_dict('records'))
    diff_types=['full_grid_range','material_simple_difference','climate_simple_difference','selected_endpoint_cell_difference']
    q=t3[t3.object_type.isin(diff_types)]
    assert len(q)>0 and set(q.metric)=={'Difference in Spearman rho'}

    t5=pd.read_csv(root/'results/reference/tables/table_5_temporal_and_robustness.csv',keep_default_na=False)
    assert set(t5.panel)=={'A_grid_level'}
    assert set(t5.panel_heading)=={'Panel A — Grid-level robustness and sensitivity'}
    assert set(t5.analytical_object)=={'Full-grid range'}
    assert set(t5.metric)=={'Difference in Spearman rho'}
    assert not t5.public_label.astype(str).str.contains('endpoint',case=False).any()
    assert not t5.astype(str).apply(lambda c:c.str.contains('primary full-grid robustness',case=False,regex=False)).any().any()
    q=t5[t5.diagnostic_key.eq('rmc_exclude_mixed_changed')]
    assert len(q)==1 and q.iloc[0].scope_or_n=='n = 20'
    assert 'mixed/changed' in q.iloc[0].interpretation.lower()

    s5=pd.read_csv(root/'supplementary_tables/table_s5_extended_robustness.csv',keep_default_na=False)
    a=s5[s5.panel.eq('A_selected_endpoint_sensitivity')]
    assert len(a)>=8 and 'primary_reference' not in set(a.evidence_class)
    assert 'primary' not in ' '.join(a[['evidence_class','interpretation']].astype(str).to_numpy().ravel()).lower()
    b=s5[s5.panel.eq('B_trend_interval_summary')]
    counts=pd.read_csv(root/'supplementary_data/trend_interval_counts.csv',keep_default_na=False)
    assert len(b)==len(counts)==20
    got=b[['representation','method','interval_excludes_zero_n','total_countries']].copy().sort_values(['representation','method']).reset_index(drop=True)
    exp=counts[['representation','method','interval_excludes_zero_n','total_countries']].copy().sort_values(['representation','method']).reset_index(drop=True)
    assert (got[['representation','method']].to_numpy()==exp[['representation','method']].to_numpy()).all()
    for c in ['interval_excludes_zero_n','total_countries']:
        assert np.allclose(pd.to_numeric(got[c]).to_numpy(float),pd.to_numeric(exp[c]).to_numpy(float),rtol=0,atol=0)

def main():
    m=load_renderer()
    td=Path(tempfile.mkdtemp(prefix="publication_tables_"))
    try:
        (td/"data").mkdir(); shutil.copy2(ROOT/"data"/"sources.csv",td/"data"/"sources.csv"); shutil.copy2(ROOT/"data"/"data_dictionary.csv",td/"data"/"data_dictionary.csv")
        m.build_publication_sources(ROOT/"results"/"reference",td)
        for fn in EXPECTED_MAIN: compare(td/"results"/"reference"/"tables"/fn,ROOT/"results"/"reference"/"tables"/fn)
        supp=list((ROOT/"supplementary_tables").glob("table_s*.csv")); assert len(supp)==8
        for p in supp: compare(td/"supplementary_tables"/p.name,p)
        semantic_assertions(td)
        print("PUBLICATION_TABLES_TEST_PASS main=4 supplementary=8 tolerance=1e-12 semantics=PASS")
    finally: shutil.rmtree(td,ignore_errors=True)
if __name__=="__main__": main()
