#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from checks import publication_relpaths

BASE={
'a_CMUR_TERR':0.09218559218559218,
'b_DMC_TERR':0.4065934065934066,
'c_RMC_TERR':0.5366300366300367,
'd_CMUR_CFOOT':0.08424908424908425,
'e_DMC_CFOOT':0.4719169719169719,
'f_RMC_CFOOT':0.688034188034188,
'full_grid_range':0.6037851037851037,
}
EVENT={
'a_CMUR_TERR':0.115995115995116,
'b_DMC_TERR':0.360805860805861,
'c_RMC_TERR':0.528083028083028,
'd_CMUR_CFOOT':0.094017094017094,
'e_DMC_CFOOT':0.419413919413919,
'f_RMC_CFOOT':0.647130647130647,
'full_grid_range':0.553113553113553,
}
FLAGGED={'BE','CZ','DE','FR','HU','LU','LV','PL','PT','RO'}

def close(a,b,t=1e-12): return abs(float(a)-float(b))<=t

def main():
    ext=ROOT/'results/reference/supplementary_data'
    fixed=pd.read_csv(ext/'full_grid_fixed_sensitivities.csv',keep_default_na=False).set_index('diagnostic_key')
    for c,v in BASE.items(): assert close(fixed.loc['baseline',c],v),(c,fixed.loc['baseline',c],v)
    for c,v in EVENT.items(): assert close(fixed.loc['exclude_2020_2021',c],v),(c,fixed.loc['exclude_2020_2021',c],v)
    ev=fixed.loc['exclude_2020_2021']
    assert int(ev.n_years)==12 and ev.specification=='primary_common_pc' and ev.estimator=='OLS' and ev.cmur_mode=='pp'
    assert ev.full_grid_minimum_cell_key=='d_CMUR_CFOOT' and ev.full_grid_maximum_cell_key=='f_RMC_CFOOT'

    paired=pd.read_csv(ext/'bootstrap_paired_contrast_summary.csv',keep_default_na=False)
    assert len(paired)==18 and paired.groupby(['contrast_id','block_length']).size().eq(1).all()
    assert set(paired.block_length.astype(int))=={2,3} and set(paired.B.astype(int))=={10000}
    assert len(set(paired.contrast_id))==9
    neg=paired[paired.contrast_id.eq('consumption_minus_territorial_cmur')]
    assert len(neg)==2 and (neg.observed_difference.astype(float)<0).all()
    for cid in ['dmc_minus_cmur_territorial','dmc_minus_cmur_consumption','rmc_minus_cmur_territorial','rmc_minus_cmur_consumption']:
        assert (paired.loc[paired.contrast_id.eq(cid),'diagnostic_q025'].astype(float)>0).all(),cid
    q=paired[paired.contrast_id.eq('rmc_minus_dmc_territorial')]
    assert (q.diagnostic_q025.astype(float)<0).all() and (q.diagnostic_q975.astype(float)>0).all()
    q=paired[paired.contrast_id.eq('rmc_minus_dmc_consumption')]; assert (q.diagnostic_q025.astype(float)>0).all()
    for cid in ['consumption_minus_territorial_cmur','consumption_minus_territorial_dmc']:
        q=paired[paired.contrast_id.eq(cid)]; assert (q.diagnostic_q025.astype(float)<0).all() and (q.diagnostic_q975.astype(float)>0).all()
    q=paired[paired.contrast_id.eq('consumption_minus_territorial_rmc')]; assert (q.diagnostic_q025.astype(float)>0).all()
    gs=pd.read_csv(ext/'bootstrap_grid_summary.csv')
    for L in [2,3]:
        q=gs[gs.block_length.eq(L)].set_index('quantity')
        pm=float(paired[(paired.block_length.eq(L))&(paired.contrast_id.eq('dmc_minus_cmur_territorial'))].iloc[0].bootstrap_median)
        # A within-draw median need not equal a subtraction of marginal medians; in this fixed run it demonstrably does not.
        assert abs(pm-(float(q.loc['b_DMC_TERR','median'])-float(q.loc['a_CMUR_TERR','median'])))>1e-6

    pop=pd.read_csv(ext/'population_flag_rank_displacement_summary.csv',keep_default_na=False)
    assert len(pop)==6 and pop.groupby(['comparison_key','flag_group']).size().eq(1).all()
    for countries in pop.loc[pop.flag_group.eq('at_least_one_population_flag'),'countries']:
        assert set(countries.split(';'))==FLAGGED
    exp={('CMUR_to_DMC','at_least_one_population_flag'):8.3,('CMUR_to_DMC','no_population_flag'):7.235294117647,
         ('DMC_to_RMC','at_least_one_population_flag'):3.7,('DMC_to_RMC','no_population_flag'):3.235294117647,
         ('TERR_to_CFOOT','at_least_one_population_flag'):2.5,('TERR_to_CFOOT','no_population_flag'):3.588235294118}
    for (k,g),v in exp.items():
        r=pop[(pop.comparison_key.eq(k))&(pop.flag_group.eq(g))]; assert len(r)==1 and close(r.iloc[0].mean_abs_rank_displacement,v)
    assert not pop.interpretation.str.contains(r'no effect|do not matter|statistically unrelated',case=False,regex=True).any()

    # Traceability and inventory hard gates.
    counts={}
    for fn in sorted((ROOT/'supplementary_tables').glob('table_s*.csv')):
        counts[fn.name]=len(pd.read_csv(fn,keep_default_na=False))
    assert counts=={
      'table_s1_sources_units_provenance.csv':12,
      'table_s2_temporal_concordance_matrices.csv':18,
      'table_s3_companion_correlations.csv':36,
      'table_s4_temporal_ordered_contrasts.csv':42,
      'table_s5_extended_robustness.csv':41,
      'table_s6_temporal_bootstrap_diagnostics.csv':30,
      'table_s7_production_account_bridge.csv':9,
      'table_s8_rmc_provenance.csv':39,
      'table_s9_source_denominator_tie_scale_diagnostics.csv':46,
    }
    s5=pd.read_csv(ROOT/'supplementary_tables/table_s5_extended_robustness.csv',keep_default_na=False)
    assert s5.groupby('panel').size().to_dict()=={'A_grid_level':11,'B_selected_endpoint':10,'C_trend_intervals':20}
    s6=pd.read_csv(ROOT/'supplementary_tables/table_s6_temporal_bootstrap_diagnostics.csv',keep_default_na=False)
    assert s6.groupby('panel').size().to_dict()=={'A_cellwise':12,'B_paired_contrasts':18}
    assert s6.interpretation.str.contains('do not propagate uncertainty from the upstream statistical-production models',regex=False).all()
    s8=pd.read_csv(ROOT/'supplementary_tables/table_s8_rmc_provenance.csv',keep_default_na=False)
    assert s8.groupby('panel').size().to_dict()=={'A_country_registry':27,'B_rmc_provenance_sensitivity':3,'C_registry_groups':9}
    s9=pd.read_csv(ROOT/'supplementary_tables/table_s9_source_denominator_tie_scale_diagnostics.csv',keep_default_na=False)
    assert s9.groupby('panel').size().to_dict()=={'A_source_status':14,'B_published_vs_common':12,'C_tie_groups':12,'D_selected_scale':2,'E_population_any_flag':6}
    assert len(list(ext.iterdir()))==30
    assert len(list((ROOT/'figures/source_data').glob('*.csv')))==13
    assert len([p for p in (ROOT/'figures/reference').iterdir() if p.suffix in {'.pdf','.png','.svg'}])==39
    assert len(publication_relpaths())==96
    stale=['table_s6_production_account_bridge.csv','table_s7_rmc_status_provenance.csv','table_s8_numeric_scale_diagnostics.csv']
    assert not any((ROOT/'supplementary_tables'/x).exists() for x in stale)

    t5=pd.read_csv(ROOT/'results/reference/tables/table_5_temporal_and_robustness.csv',keep_default_na=False)
    a=s5[s5.panel.eq('A_grid_level')]
    shared=['panel','panel_heading','evidence_class','diagnostic_key','public_label','scope_or_n','analytical_object','metric','reference_or_point','lower_value','upper_value','minimum_cell_context','maximum_cell_context','interpretation']
    x=t5[shared].reset_index(drop=True); y=a[shared].reset_index(drop=True)
    assert x.shape==y.shape
    for c in shared:
        xn=pd.to_numeric(x[c],errors='coerce'); yn=pd.to_numeric(y[c],errors='coerce')
        if xn.notna().sum()==len(x) and yn.notna().sum()==len(y):
            assert np.allclose(xn.to_numpy(float),yn.to_numpy(float),rtol=0,atol=1e-12),c
        else:
            assert (x[c].astype(str).to_numpy()==y[c].astype(str).to_numpy()).all(),c
    print('PUBLICATION_EVIDENCE_CONTRACT_TEST_PASS baseline=UNCHANGED event=PASS paired=18 population=6 S4_S9=PASS publication_relpaths=96')

if __name__=='__main__': main()
