#!/usr/bin/env python3
from pathlib import Path
import importlib.util, shutil, tempfile
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXPECTED_MAIN=[
"table_2_representation_contract_source.csv","table_3_primary_concordance_and_contrasts.csv","table_4_country_consequences.csv","table_5_temporal_and_robustness.csv"]
EXPECTED_SUPP=[
"table_s1_sources_units_provenance.csv","table_s2_temporal_concordance_matrices.csv","table_s3_companion_correlations.csv",
"table_s4_temporal_ordered_contrasts.csv","table_s5_extended_robustness.csv","table_s6_temporal_bootstrap_diagnostics.csv",
"table_s7_production_account_bridge.csv","table_s8_rmc_provenance.csv","table_s9_source_denominator_tie_scale_diagnostics.csv"]

def load_renderer():
    p=ROOT/"code"/"04_render_publication_figures.py"; spec=importlib.util.spec_from_file_location("publication_renderer",p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def compare(a,b,tol=1e-12):
    x=pd.read_csv(a,keep_default_na=False); y=pd.read_csv(b,keep_default_na=False)
    assert list(x.columns)==list(y.columns) and x.shape==y.shape,(a,b,x.shape,y.shape)
    for c in x.columns:
        xn=pd.to_numeric(x[c],errors="coerce"); yn=pd.to_numeric(y[c],errors="coerce")
        if xn.notna().sum()==len(x) and yn.notna().sum()==len(y):
            if len(x): assert np.max(np.abs(xn.to_numpy(float)-yn.to_numpy(float)))<=tol,c
        else: assert (x[c].astype(str).to_numpy()==y[c].astype(str).to_numpy()).all(),c

def semantic_assertions(root):
    t3=pd.read_csv(root/'results/reference/tables/table_3_primary_concordance_and_contrasts.csv',keep_default_na=False)
    for typ in ['full_grid_minimum','full_grid_maximum']:
        q=t3[t3.object_type.eq(typ)]; assert len(q)==1 and q.iloc[0].metric=='Spearman rho'
    q=t3[t3.object_type.isin(['full_grid_range','material_simple_difference','climate_simple_difference','selected_endpoint_cell_difference'])]
    assert len(q)>0 and set(q.metric)=={'Difference in Spearman rho'}

    t5=pd.read_csv(root/'results/reference/tables/table_5_temporal_and_robustness.csv',keep_default_na=False)
    assert len(t5)==11 and set(t5.panel)=={'A_grid_level'} and set(t5.panel_heading)=={'Panel A — Grid-level robustness and sensitivity'}
    assert set(t5.analytical_object)=={'Full-grid range'} and set(t5.metric)=={'Difference in Spearman rho'}
    q=t5[t5.diagnostic_key.eq('exclude_2020_2021')]; assert len(q)==1
    assert q.iloc[0].scope_or_n=='n = 27; 12 retained calendar years' and abs(float(q.iloc[0].reference_or_point)-0.553113553113553)<=1e-12
    assert q.iloc[0].minimum_cell_context=='CMUR × consumption' and q.iloc[0].maximum_cell_context=='RMC × consumption'
    assert q.iloc[0].interpretation=='All five trajectories re-estimated after excluding calendar years 2020 and 2021.'

    s4=pd.read_csv(root/'supplementary_tables/table_s4_temporal_ordered_contrasts.csv',keep_default_na=False)
    assert len(s4)==42 and (s4.record_type=='monotone_path').sum()==9
    for p in ['FULL_2010_2023','EARLY_2010_2016','LATE_2017_2023']: assert ((s4.record_type=='symmetric_temporal')&(s4.period==p)).sum()==11

    s5=pd.read_csv(root/'supplementary_tables/table_s5_extended_robustness.csv',keep_default_na=False)
    assert len(s5)==41 and (s5.panel=='A_grid_level').sum()==11 and (s5.panel=='B_selected_endpoint').sum()==10 and (s5.panel=='C_trend_intervals').sum()==20
    # Shared MAIN/S5 Panel A fields must be identical.
    cols=['panel','panel_heading','evidence_class','diagnostic_key','public_label','scope_or_n','analytical_object','metric','reference_or_point','lower_value','upper_value','minimum_cell_context','maximum_cell_context','interpretation']
    a=s5[s5.panel.eq('A_grid_level')][cols].reset_index(drop=True); b=t5[cols].reset_index(drop=True)
    assert a.shape==b.shape
    for c in cols:
        an=pd.to_numeric(a[c],errors='coerce'); bn=pd.to_numeric(b[c],errors='coerce')
        if an.notna().sum()==len(a) and bn.notna().sum()==len(b): assert np.allclose(an,bn,rtol=0,atol=1e-12)
        else: assert (a[c].astype(str).to_numpy()==b[c].astype(str).to_numpy()).all(),c

    s6=pd.read_csv(root/'supplementary_tables/table_s6_temporal_bootstrap_diagnostics.csv',keep_default_na=False)
    assert len(s6)==30 and (s6.panel=='A_cellwise').sum()==12 and (s6.panel=='B_paired_contrasts').sum()==18
    assert s6.interpretation.str.contains('do not propagate uncertainty from the upstream statistical-production models',regex=False).all()
    assert s6[s6.panel.eq('B_paired_contrasts')].groupby(['object_id','block_length']).size().eq(1).all()

    s7=pd.read_csv(root/'supplementary_tables/table_s7_production_account_bridge.csv',keep_default_na=False); assert len(s7)==9
    s8=pd.read_csv(root/'supplementary_tables/table_s8_rmc_provenance.csv',keep_default_na=False)
    assert len(s8)==39 and (s8.panel=='A_country_registry').sum()==27 and (s8.panel=='B_rmc_provenance_sensitivity').sum()==3 and (s8.panel=='C_registry_groups').sum()==9
    s9=pd.read_csv(root/'supplementary_tables/table_s9_source_denominator_tie_scale_diagnostics.csv',keep_default_na=False)
    exp={'A_source_status':14,'B_published_vs_common':12,'C_tie_groups':12,'D_selected_scale':2,'E_population_any_flag':6}
    assert len(s9)==46 and s9.groupby('panel').size().to_dict()==exp

def main():
    m=load_renderer(); td=Path(tempfile.mkdtemp(prefix='publication_tables_'))
    try:
        (td/'data').mkdir(); shutil.copy2(ROOT/'data/sources.csv',td/'data/sources.csv'); shutil.copy2(ROOT/'data/data_dictionary.csv',td/'data/data_dictionary.csv')
        m.build_publication_sources(ROOT/'results/reference',td)
        for fn in EXPECTED_MAIN: compare(td/'results/reference/tables'/fn,ROOT/'results/reference/tables'/fn)
        supp=sorted((ROOT/'supplementary_tables').glob('table_s*.csv')); assert [p.name for p in supp]==EXPECTED_SUPP
        for p in supp: compare(td/'supplementary_tables'/p.name,p)
        semantic_assertions(td)
        print('PUBLICATION_TABLES_TEST_PASS main=4 supplementary=9 tolerance=1e-12 semantics=PASS')
    finally: shutil.rmtree(td,ignore_errors=True)
if __name__=='__main__': main()
