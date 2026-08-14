#!/usr/bin/env python3
from pathlib import Path
import json, subprocess, sys, tempfile, shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FILES=[
    'symmetric_grid_and_temporal_effects.csv','all_monotone_paths.csv','within_axis_concordance.csv',
    'country_point_sign_and_loyo_stability.csv','point_sign_disagreement_loyo_cases.csv','point_sign_disagreement_loyo_summary.csv',
    'theil_sen_country_signs.csv','theil_sen_disagreement_cases.csv','theil_sen_disagreement_summary.csv',
    'trend_interval_diagnostics.csv','trend_interval_counts.csv','source_status_audit.csv',
    'rmc_country_method_registry.csv','rmc_official_evidence_register.csv','bootstrap_config.json','bootstrap_grid_summary.csv',
    'bootstrap_endpoint_summary.csv','bootstrap_country_rank_diagnostic_intervals.csv','bootstrap_slope_favorable_frequencies.csv',
    'bootstrap_sign_disagreement_frequencies.csv','bootstrap_rank_displacement_diagnostics.csv','bootstrap_grid_extreme_identity_frequencies.csv',
    'bootstrap_full_grid_robustness.csv','full_grid_fixed_sensitivities.csv','full_grid_deletion_runs.csv',
    'rmc_provenance_sensitivity_n20.csv','rmc_provenance_group_grids.csv','software_and_method.json']

def compare_csv(a,b,tol=1e-12):
    x=pd.read_csv(a,keep_default_na=False); y=pd.read_csv(b,keep_default_na=False)
    assert list(x.columns)==list(y.columns) and x.shape==y.shape,(Path(a).name,x.shape,y.shape)
    for c in x.columns:
        xn=pd.to_numeric(x[c],errors='coerce'); yn=pd.to_numeric(y[c],errors='coerce')
        if xn.notna().sum()==len(x) and yn.notna().sum()==len(y):
            if len(x): assert float(np.max(np.abs(xn.to_numpy(float)-yn.to_numpy(float))))<=tol,(Path(a).name,c)
        else:
            assert (x[c].astype(str).to_numpy()==y[c].astype(str).to_numpy()).all(),(Path(a).name,c)

def main():
    td=Path(tempfile.mkdtemp(prefix='extended_evidence_'))
    try:
        analysis=td/'analysis'
        subprocess.run([sys.executable,str(ROOT/'code/05_build_extended_publication_evidence.py'),
                        '--panel',str(ROOT/'data/panel_2010_2023.csv'),'--analysis',str(analysis),'--bootstrap-b','10000'],check=True,cwd=ROOT)
        got=analysis/'supplementary_data'; ref=ROOT/'results/reference/supplementary_data'
        assert sorted(p.name for p in got.iterdir())==sorted(FILES)
        for name in FILES:
            a=got/name; b=ref/name; assert a.is_file() and b.is_file(),name
            if name.endswith('.csv'): compare_csv(a,b)
            else: assert json.loads(a.read_text())==json.loads(b.read_text()),name

        sym=pd.read_csv(got/'symmetric_grid_and_temporal_effects.csv')
        q=sym[(sym.period=='FULL_2010_2023')&(sym.object_type=='full_grid_range')]
        assert len(q)==1 and abs(float(q.iloc[0].value)-0.6037851037851037)<=1e-12
        summ=pd.read_csv(got/'point_sign_disagreement_loyo_summary.csv').set_index('comparison_key')
        expected={'CMUR_to_DMC':(13,11),'DMC_to_RMC':(6,3),'TERR_to_CFOOT':(5,2)}
        for k,(n,s) in expected.items():
            assert int(summ.loc[k,'point_slope_sign_disagreements_n'])==n
            assert int(summ.loc[k,'both_sign_loyo_stable_n'])==s
        ts=pd.read_csv(got/'theil_sen_disagreement_summary.csv').set_index('comparison_key')
        for k,(n,s) in {'CMUR_to_DMC':(13,13),'DMC_to_RMC':(6,5),'TERR_to_CFOOT':(5,5)}.items():
            assert int(ts.loc[k,'ols_disagreement_n'])==n
            assert int(ts.loc[k,'theil_sen_persistence_n'])==s
        ep=pd.read_csv(got/'bootstrap_endpoint_summary.csv').set_index('block_length')
        assert abs(float(ep.loc[2,'median'])-0.6001221001221001)<=1e-12
        assert abs(float(ep.loc[2,'diagnostic_q025'])-0.4786324786324786)<=1e-12
        assert abs(float(ep.loc[2,'diagnostic_q975'])-0.7240689865689864)<=1e-12
        assert float(ep.loc[2,'positive_bootstrap_frequency'])==1.0
        assert abs(float(ep.loc[3,'median'])-0.6037851037851037)<=1e-12
        assert float(ep.loc[3,'positive_bootstrap_frequency'])==1.0
        rmc=pd.read_csv(got/'rmc_country_method_registry.csv',keep_default_na=False)
        assert len(rmc)==27 and rmc.geo.nunique()==27
        for c in ['official_evidence_source','official_evidence_locator','confidence','unresolved_year_level_boundary']:
            assert c in rmc.columns,c
        for c in ['official_evidence_source','official_evidence_locator','confidence']:
            assert (rmc[c].astype(str).str.strip()!='').all(),c
        mixed=rmc['disseminated_source_regime'].astype(str).str.contains('Mixed/changed',case=False,regex=False)
        assert (rmc.loc[mixed,'unresolved_year_level_boundary'].astype(str).str.strip()!='').all(),'mixed-regime unresolved boundary must be explicit'
        assert int(mixed.sum())==7
        n20=pd.read_csv(got/'rmc_provenance_sensitivity_n20.csv',keep_default_na=False)
        assert len(n20)==1 and int(n20.iloc[0].n_countries)==20
        assert set(n20.iloc[0].excluded_countries.split(';'))==set(rmc.loc[mixed,'geo'])
        groups=pd.read_csv(got/'rmc_provenance_group_grids.csv',keep_default_na=False).set_index('provenance_group')
        assert {k:int(groups.loc[k,'n_countries']) for k in groups.index}=={'country-reported':9,'Eurostat-estimated':11,'mixed/changed':7}
        deletion=pd.read_csv(got/'full_grid_deletion_runs.csv',keep_default_na=False)
        assert (deletion.full_grid_minimum_cell_key.astype(str).str.strip()!='').all()
        assert (deletion.full_grid_maximum_cell_key.astype(str).str.strip()!='').all()
        assert len(deletion[deletion.deletion_type=='LOCO'])==27 and len(deletion[deletion.deletion_type=='LOYO'])==14
        bootfg=pd.read_csv(got/'bootstrap_full_grid_robustness.csv',keep_default_na=False)
        assert set(bootfg.block_length.astype(int))=={2,3}
        status=pd.read_csv(got/'source_status_audit.csv')
        nonblank=status[status.status_code!='BLANK'].groupby('source')['count'].sum().to_dict()
        assert nonblank=={'CMUR':1,'DMC':22,'RMC':172,'population_denominator':30}
        print(f'EXTENDED_PUBLICATION_EVIDENCE_TEST_PASS files={len(FILES)} bootstrap_B_per_block=10000 full_grid_semantics=PASS')
    finally:
        shutil.rmtree(td,ignore_errors=True)
if __name__=='__main__': main()
