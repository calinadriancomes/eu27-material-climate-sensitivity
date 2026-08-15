#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, json, math, hashlib, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, theilslopes, rankdata
import statsmodels.api as sm

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from checks import EU27, YEARS, PERIODS, EVEN_YEARS, RANK_DECIMALS, PANEL_SHA256, compare_panel

REP_LABELS=["CMUR","DMC","RMC","TERR","CFOOT"]
PRIMARY_COLS={
    "CMUR":"cmur_pct",
    "DMC":"dmc_pc_common_t",
    "RMC":"rmc_pc_common_t",
    "TERR":"terr_ghg_pc_common_t",
    "CFOOT":"cfoot_pc_common_t",
}
SPEC_COLS={
    "primary_common_pc": PRIMARY_COLS,
    "published_pc": {
        "CMUR":"cmur_pct", "DMC":"dmc_pc_published_t", "RMC":"rmc_pc_published_t",
        "TERR":"terr_ghg_pc_published_t", "CFOOT":"cfoot_pc_published_t",
    },
    "total_scale": {
        "CMUR":"cmur_pct", "DMC":"dmc_total_t", "RMC":"rmc_total_t",
        "TERR":"terr_ghg_total_t", "CFOOT":"cfoot_total_t",
    },
}
GRID_KEYS=["a_CMUR_TERR","b_DMC_TERR","c_RMC_TERR","d_CMUR_CFOOT","e_DMC_CFOOT","f_RMC_CFOOT"]
GRID_LABELS={
    "a_CMUR_TERR":"CMUR × territorial", "b_DMC_TERR":"DMC × territorial",
    "c_RMC_TERR":"RMC × territorial", "d_CMUR_CFOOT":"CMUR × consumption",
    "e_DMC_CFOOT":"DMC × consumption", "f_RMC_CFOOT":"RMC × consumption",
}
EXCLUDE_2020_2021_YEARS=[y for y in YEARS if y not in (2020,2021)]
CONTRAST_DEFS=[
    ("dmc_minus_cmur_territorial","DMC − CMUR | territorial","b_DMC_TERR","a_CMUR_TERR"),
    ("dmc_minus_cmur_consumption","DMC − CMUR | consumption","e_DMC_CFOOT","d_CMUR_CFOOT"),
    ("rmc_minus_dmc_territorial","RMC − DMC | territorial","c_RMC_TERR","b_DMC_TERR"),
    ("rmc_minus_dmc_consumption","RMC − DMC | consumption","f_RMC_CFOOT","e_DMC_CFOOT"),
    ("rmc_minus_cmur_territorial","RMC − CMUR | territorial","c_RMC_TERR","a_CMUR_TERR"),
    ("rmc_minus_cmur_consumption","RMC − CMUR | consumption","f_RMC_CFOOT","d_CMUR_CFOOT"),
    ("consumption_minus_territorial_cmur","Consumption − territorial | CMUR","d_CMUR_CFOOT","a_CMUR_TERR"),
    ("consumption_minus_territorial_dmc","Consumption − territorial | DMC","e_DMC_CFOOT","b_DMC_TERR"),
    ("consumption_minus_territorial_rmc","Consumption − territorial | RMC","f_RMC_CFOOT","c_RMC_TERR"),
]


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def transform_series(v, rep, cmur_mode="pp"):
    v=np.asarray(v,float)
    if rep=="CMUR":
        if cmur_mode=="pp": return v
        if cmur_mode=="log":
            if np.any(v<=0): raise ValueError("non-positive CMUR")
            return np.log(v)
        raise ValueError(cmur_mode)
    if np.any(v<=0): raise ValueError(f"non-positive {rep}")
    return -np.log(v)


def ols_slope(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); xc=x-x.mean()
    return float(np.dot(xc,y)/np.dot(xc,xc))


def progress_table(panel, geos, years, estimator="OLS", spec="primary_common_pc", cmur_mode="pp"):
    if spec not in SPEC_COLS: raise ValueError(spec)
    cols=SPEC_COLS[spec]
    rows=[]
    for geo in geos:
        g=panel[(panel.geo==geo)&panel.year.isin(years)].sort_values('year')
        if list(g.year.astype(int))!=list(years): raise RuntimeError(f"{geo}: incomplete annual support")
        x=g.year.to_numpy(float); row={'geo':geo}
        for rep,col in cols.items():
            y=transform_series(g[col].to_numpy(float),rep,cmur_mode if rep=="CMUR" else "pp")
            if estimator=="OLS": val=ols_slope(x,y)
            elif estimator=="THEIL_SEN": val=float(theilslopes(y,x,method='separate')[0])
            else: raise ValueError(estimator)
            row[rep]=val
        rows.append(row)
    return pd.DataFrame(rows)


def canon(a): return np.round(np.asarray(a,float),RANK_DECIMALS)

def corr(a,b,method='spearman',canonical=True):
    x=np.asarray(a,float); y=np.asarray(b,float)
    if canonical and method in ('spearman','kendall'): x,y=canon(x),canon(y)
    if method=='spearman': return float(spearmanr(x,y).statistic)
    if method=='kendall': return float(kendalltau(x,y,variant='b').statistic)
    if method=='pearson': return float(np.corrcoef(x,y)[0,1])
    raise ValueError(method)


def grid(pr):
    a=corr(pr.CMUR,pr.TERR); b=corr(pr.DMC,pr.TERR); c=corr(pr.RMC,pr.TERR)
    d=corr(pr.CMUR,pr.CFOOT); e=corr(pr.DMC,pr.CFOOT); f=corr(pr.RMC,pr.CFOOT)
    return {'a_CMUR_TERR':a,'b_DMC_TERR':b,'c_RMC_TERR':c,'d_CMUR_CFOOT':d,'e_DMC_CFOOT':e,'f_RMC_CFOOT':f,
            'selected_endpoint_difference':f-a}



def grid_record(g, **meta):
    cells={k:float(g[k]) for k in GRID_KEYS}
    mn=min(GRID_KEYS,key=lambda k:cells[k]); mx=max(GRID_KEYS,key=lambda k:cells[k])
    rec=dict(meta)
    rec.update(cells)
    rec.update({'cell_metric':'Spearman rho','full_grid_range_metric':'Difference in Spearman rho','selected_endpoint_metric':'Difference in Spearman rho'})
    rec.update({
        'full_grid_minimum':cells[mn], 'full_grid_minimum_cell_key':mn, 'full_grid_minimum_cell_label':GRID_LABELS[mn],
        'full_grid_maximum':cells[mx], 'full_grid_maximum_cell_key':mx, 'full_grid_maximum_cell_label':GRID_LABELS[mx],
        'full_grid_range':cells[mx]-cells[mn],
        'selected_endpoint_cell_difference':cells['f_RMC_CFOOT']-cells['a_CMUR_TERR'],
        'dmc_minus_cmur_territorial':cells['b_DMC_TERR']-cells['a_CMUR_TERR'],
        'dmc_minus_cmur_consumption':cells['e_DMC_CFOOT']-cells['d_CMUR_CFOOT'],
        'rmc_minus_dmc_territorial':cells['c_RMC_TERR']-cells['b_DMC_TERR'],
        'rmc_minus_dmc_consumption':cells['f_RMC_CFOOT']-cells['e_DMC_CFOOT'],
        'rmc_minus_cmur_territorial':cells['c_RMC_TERR']-cells['a_CMUR_TERR'],
        'rmc_minus_cmur_consumption':cells['f_RMC_CFOOT']-cells['d_CMUR_CFOOT'],
        'consumption_minus_territorial_cmur':cells['d_CMUR_CFOOT']-cells['a_CMUR_TERR'],
        'consumption_minus_territorial_dmc':cells['e_DMC_CFOOT']-cells['b_DMC_TERR'],
        'consumption_minus_territorial_rmc':cells['f_RMC_CFOOT']-cells['c_RMC_TERR'],
    })
    return rec


def full_grid_authority(panel, registry):
    fixed=[]
    fixed_specs=[
        ('baseline','primary_common_pc','OLS','pp',YEARS,'Full 2010–2023 baseline'),
        ('published_pc','published_pc','OLS','pp',YEARS,'Published per-capita convention'),
        ('common_log_CMUR','primary_common_pc','OLS','log',YEARS,'Common-log CMUR convention'),
        ('Theil_Sen','primary_common_pc','THEIL_SEN','pp',YEARS,'Theil–Sen estimator'),
        ('total_scale','total_scale','OLS','pp',YEARS,'National-total scale convention'),
        ('common_even_year','primary_common_pc','OLS','pp',EVEN_YEARS,'Common even-year support'),
        ('exclude_2020_2021','primary_common_pc','OLS','pp',EXCLUDE_2020_2021_YEARS,'Exclude 2020 and 2021'),
    ]
    for key,spec,est,mode,years,label in fixed_specs:
        pr=progress_table(panel,EU27,years,estimator=est,spec=spec,cmur_mode=mode)
        fixed.append(grid_record(grid(pr), diagnostic_key=key, public_label=label, n_countries=len(EU27), n_years=len(years), specification=spec, estimator=est, cmur_mode=mode))

    deletion=[]
    deletion.append(grid_record(grid(progress_table(panel,EU27,YEARS)), deletion_type='FULL', deletion_id='FULL_2010_2023', n_countries=27, n_years=14))
    for geo in EU27:
        geos=[g for g in EU27 if g!=geo]
        deletion.append(grid_record(grid(progress_table(panel,geos,YEARS)), deletion_type='LOCO', deletion_id=geo, n_countries=26, n_years=14))
    for year in YEARS:
        years=[y for y in YEARS if y!=year]
        deletion.append(grid_record(grid(progress_table(panel,EU27,years)), deletion_type='LOYO', deletion_id=str(year), n_countries=27, n_years=13))

    if len(registry)!=27 or registry.geo.nunique()!=27: raise RuntimeError('RMC provenance registry must contain exactly 27 countries')
    regimes=registry.set_index('geo')['disseminated_source_regime'].astype(str)
    mixed=sorted(regimes[regimes.eq('Mixed/changed over 2010–2023')].index.tolist())
    if len(mixed)!=7: raise RuntimeError(f'expected 7 mixed/changed RMC cases, found {len(mixed)}: {mixed}')
    included=[g for g in EU27 if g not in mixed]
    if len(included)!=20: raise RuntimeError('RMC mixed/changed exclusion did not yield n=20')
    n20=grid_record(grid(progress_table(panel,included,YEARS)),
                    sensitivity_rule='Exclude exactly the frozen-registry mixed/changed cases',
                    n_countries=20, included_countries=';'.join(included), excluded_countries=';'.join(mixed),
                    composition_caveat='Provenance regime and country composition are confounded; this is a provenance/composition sensitivity, not a production-method effect.')

    groups=[]
    group_map=[('country-reported','Country-reported'),('Eurostat-estimated','Eurostat-estimated'),('mixed/changed','Mixed/changed over 2010–2023')]
    for key,regime in group_map:
        geos=[g for g in EU27 if regimes.loc[g]==regime]
        expected={'country-reported':9,'Eurostat-estimated':11,'mixed/changed':7}[key]
        if len(geos)!=expected: raise RuntimeError(f'{key} membership mismatch: {len(geos)} != {expected}')
        groups.append(grid_record(grid(progress_table(panel,geos,YEARS)), provenance_group=key, registry_regime=regime, n_countries=len(geos), countries=';'.join(geos),
                                  composition_caveat='Group comparisons confound provenance regime with country composition and are not production-method effects.'))
    return pd.DataFrame(fixed),pd.DataFrame(deletion),pd.DataFrame([n20]),pd.DataFrame(groups)


def bootstrap_full_grid_summary(grid_summary, extreme_identities):
    rows=[]
    for L in sorted(grid_summary.block_length.unique()):
        q=grid_summary[grid_summary.block_length.eq(L)].set_index('quantity')
        cells={k:float(q.loc[k,'median']) for k in GRID_KEYS}
        mn=min(GRID_KEYS,key=lambda k:cells[k]); mx=max(GRID_KEYS,key=lambda k:cells[k])
        ex=extreme_identities[extreme_identities.block_length.eq(L)]
        minq=ex[ex.extreme.eq('minimum')].sort_values(['frequency','cell_key'],ascending=[False,True]).iloc[0]
        maxq=ex[ex.extreme.eq('maximum')].sort_values(['frequency','cell_key'],ascending=[False,True]).iloc[0]
        rr=q.loc['full_grid_range']
        rec={'block_length':int(L),'n_countries':27,'B':int(ex.bootstrap_replicates.iloc[0]),'cell_metric':'Spearman rho','full_grid_range_metric':'Difference in Spearman rho',
             **{f'{k}_median':v for k,v in cells.items()},
             'minimum_of_cell_medians':cells[mn],'minimum_cell_median_key':mn,'minimum_cell_median_label':GRID_LABELS[mn],
             'maximum_of_cell_medians':cells[mx],'maximum_cell_median_key':mx,'maximum_cell_median_label':GRID_LABELS[mx],
             'full_grid_range_median':float(rr['median']),'full_grid_range_diagnostic_q025':float(rr['diagnostic_q025']),'full_grid_range_diagnostic_q975':float(rr['diagnostic_q975']),
             'modal_minimum_cell_key':str(minq.cell_key),'modal_minimum_cell_label':str(minq.cell_label),'modal_minimum_cell_frequency':float(minq.frequency),
             'modal_maximum_cell_key':str(maxq.cell_key),'modal_maximum_cell_label':str(maxq.cell_label),'modal_maximum_cell_frequency':float(maxq.frequency),
             'interpretation':'Joint residual-block bootstrap diagnostic; ranges are bootstrap diagnostic ranges and frequencies are bootstrap frequencies.'}
        rows.append(rec)
    return pd.DataFrame(rows)

def ranks(pr):
    z=pr.set_index('geo').copy()
    for rep in REP_LABELS:
        z[f'rank_{rep}']=pd.Series(canon(z[rep]),index=z.index).rank(method='average',ascending=False)
        z[f'sign_{rep}']=np.where(z[rep]>0,'POSITIVE_FAVORABLE','NONPOSITIVE_NONFAVORABLE')
    return z.reset_index()


def consequences(pr):
    z=ranks(pr).set_index('geo'); out=pd.DataFrame(index=z.index)
    for before,after,key in [('CMUR','DMC','CMUR_to_DMC'),('DMC','RMC','DMC_to_RMC'),('TERR','CFOOT','TERR_to_CFOOT')]:
        out[f'shift_{key}']=z[f'rank_{after}']-z[f'rank_{before}']
        out[f'disagree_{key}']=(z[before]>0)!=(z[after]>0)
    return z.join(out).reset_index()


def symmetric_effects(panel):
    rows=[]
    cellnames={'a_CMUR_TERR':'CMUR × territorial','b_DMC_TERR':'DMC × territorial','c_RMC_TERR':'RMC × territorial',
               'd_CMUR_CFOOT':'CMUR × consumption','e_DMC_CFOOT':'DMC × consumption','f_RMC_CFOOT':'RMC × consumption'}
    for period,years in PERIODS.items():
        g=grid(progress_table(panel,EU27,years)); cells={k:g[k] for k in cellnames}
        for k,label in cellnames.items(): rows.append({'period':period,'object_type':'grid_cell','effect':label,'value':g[k],'formula':k})
        mn=min(cells,key=cells.get); mx=max(cells,key=cells.get)
        rows.append({'period':period,'object_type':'full_grid_minimum','effect':cellnames[mn],'value':cells[mn],'formula':mn})
        rows.append({'period':period,'object_type':'full_grid_maximum','effect':cellnames[mx],'value':cells[mx],'formula':mx})
        rows.append({'period':period,'object_type':'full_grid_range','effect':'maximum minus minimum across six cells','value':cells[mx]-cells[mn],'formula':f'{mx} - {mn}'})
        defs=[
          ('material_simple_difference','DMC − CMUR | territorial',g['b_DMC_TERR']-g['a_CMUR_TERR'],'b-a'),
          ('material_simple_difference','DMC − CMUR | consumption',g['e_DMC_CFOOT']-g['d_CMUR_CFOOT'],'e-d'),
          ('material_simple_difference','RMC − DMC | territorial',g['c_RMC_TERR']-g['b_DMC_TERR'],'c-b'),
          ('material_simple_difference','RMC − DMC | consumption',g['f_RMC_CFOOT']-g['e_DMC_CFOOT'],'f-e'),
          ('material_simple_difference','RMC − CMUR | territorial',g['c_RMC_TERR']-g['a_CMUR_TERR'],'c-a'),
          ('material_simple_difference','RMC − CMUR | consumption',g['f_RMC_CFOOT']-g['d_CMUR_CFOOT'],'f-d'),
          ('climate_simple_difference','consumption − territorial | CMUR',g['d_CMUR_CFOOT']-g['a_CMUR_TERR'],'d-a'),
          ('climate_simple_difference','consumption − territorial | DMC',g['e_DMC_CFOOT']-g['b_DMC_TERR'],'e-b'),
          ('climate_simple_difference','consumption − territorial | RMC',g['f_RMC_CFOOT']-g['c_RMC_TERR'],'f-c'),
        ]
        for typ,effect,val,formula in defs: rows.append({'period':period,'object_type':typ,'effect':effect,'value':val,'formula':formula})
        rows.append({'period':period,'object_type':'selected_endpoint_cell_difference','effect':'RMC × consumption minus CMUR × territorial','value':g['selected_endpoint_difference'],'formula':'f-a'})
    d=pd.DataFrame(rows); d['sign']=np.where(d.value>0,'POSITIVE',np.where(d.value<0,'NEGATIVE','ZERO'))
    return d


def monotone_paths(panel):
    g=grid(progress_table(panel,EU27,YEARS)); a,b,c,d,e,f=[g[k] for k in ['a_CMUR_TERR','b_DMC_TERR','c_RMC_TERR','d_CMUR_CFOOT','e_DMC_CFOOT','f_RMC_CFOOT']]
    paths={
      'climate_first':[('CMUR×TERR → CMUR×CFOOT',d-a),('CMUR×CFOOT → DMC×CFOOT',e-d),('DMC×CFOOT → RMC×CFOOT',f-e)],
      'climate_between_material_steps':[('CMUR×TERR → DMC×TERR',b-a),('DMC×TERR → DMC×CFOOT',e-b),('DMC×CFOOT → RMC×CFOOT',f-e)],
      'climate_last':[('CMUR×TERR → DMC×TERR',b-a),('DMC×TERR → RMC×TERR',c-b),('RMC×TERR → RMC×CFOOT',f-c)],
    }
    rows=[]; target=f-a
    for name,steps in paths.items():
        total=sum(v for _,v in steps)
        for i,(desc,val) in enumerate(steps,1):
            rows.append({'path':name,'step_number':i,'step_sequence':desc,'step_value':val,'path_sum':total,
                         'selected_endpoint_difference':target,'sum_error':total-target,'endpoint_invariance_pass':abs(total-target)<=1e-12})
    return pd.DataFrame(rows)


def within_axis(panel):
    pr=progress_table(panel,EU27,YEARS); rows=[]
    for x,y in [('CMUR','DMC'),('CMUR','RMC'),('DMC','RMC'),('TERR','CFOOT')]:
        rows.append({'axis':'material' if x in ('CMUR','DMC','RMC') and y in ('CMUR','DMC','RMC') else 'climate',
                     'comparison':f'{x} vs {y}','spearman':corr(pr[x],pr[y],'spearman',True),'kendall_tau_b':corr(pr[x],pr[y],'kendall',True),
                     'tie_rule':'12-decimal numerical canonicalization followed by average ranks'})
    return pd.DataFrame(rows)


def loyo_sign_stability(panel):
    full=progress_table(panel,EU27,YEARS); rr=ranks(full).set_index('geo')
    loyo={rep:{geo:[] for geo in EU27} for rep in REP_LABELS}
    for omit in YEARS:
        pr=progress_table(panel,EU27,[y for y in YEARS if y!=omit]).set_index('geo')
        for rep in REP_LABELS:
            for geo in EU27: loyo[rep][geo].append(float(pr.loc[geo,rep]))
    rows=[]
    for geo in EU27:
        for rep in REP_LABELS:
            v=float(rr.loc[geo,rep]); arr=np.asarray(loyo[rep][geo]); positive=v>0; preserve=(arr>0)==positive
            rows.append({'geo':geo,'representation':rep,'primary_progress_score':v,'point_sign':'POSITIVE_FAVORABLE' if positive else 'NONPOSITIVE_NONFAVORABLE',
                         'current_rank':float(rr.loc[geo,f'rank_{rep}']),'loyo_preserve_sign_n':int(preserve.sum()),'loyo_reverse_sign_n':int((~preserve).sum()),
                         'loyo_min_slope':float(arr.min()),'loyo_max_slope':float(arr.max()),'sign_stable_all_14':bool(preserve.all())})
    detail=pd.DataFrame(rows); cc=consequences(full).set_index('geo'); look=detail.set_index(['geo','representation']); cases=[]; summary=[]
    for before,after,key,label in [('CMUR','DMC','CMUR_to_DMC','CMUR → DMC'),('DMC','RMC','DMC_to_RMC','DMC → RMC'),('TERR','CFOOT','TERR_to_CFOOT','Territorial → consumption-based GHG')]:
        stable=0; total=0
        for geo in EU27:
            if not bool(cc.loc[geo,f'disagree_{key}']): continue
            total+=1; a=look.loc[(geo,before)]; b=look.loc[(geo,after)]; both=bool(a.sign_stable_all_14 and b.sign_stable_all_14); stable+=int(both)
            cases.append({'comparison':label,'comparison_key':key,'geo':geo,'first_representation':before,'second_representation':after,
                          'first_point_sign':a.point_sign,'second_point_sign':b.point_sign,
                          'first_loyo_preserve_sign_n':int(a.loyo_preserve_sign_n),'second_loyo_preserve_sign_n':int(b.loyo_preserve_sign_n),
                          'first_sign_stable_all_14':bool(a.sign_stable_all_14),'second_sign_stable_all_14':bool(b.sign_stable_all_14),'both_signs_stable_all_14':both})
        summary.append({'comparison':label,'comparison_key':key,'point_slope_sign_disagreements_n':total,'both_sign_loyo_stable_n':stable,
                        'disagreement_denominator_n':total,'both_sign_loyo_stable_fraction':stable/total})
    return detail,pd.DataFrame(cases),pd.DataFrame(summary)


def theil_sen_signs(panel):
    ols=progress_table(panel,EU27,YEARS).set_index('geo'); ts=progress_table(panel,EU27,YEARS,estimator='THEIL_SEN').set_index('geo')
    cc=consequences(ols.reset_index()).set_index('geo'); memberships={k:set(cc.index[cc[f'disagree_{k}']]) for k in ['CMUR_to_DMC','DMC_to_RMC','TERR_to_CFOOT']}
    rows=[]
    for geo in EU27:
        for rep in REP_LABELS:
            ov=float(ols.loc[geo,rep]); tv=float(ts.loc[geo,rep])
            rows.append({'geo':geo,'representation':rep,'ols_progress_score':ov,'theil_sen_progress_score':tv,
                         'ols_sign':'POSITIVE_FAVORABLE' if ov>0 else 'NONPOSITIVE_NONFAVORABLE','theil_sen_sign':'POSITIVE_FAVORABLE' if tv>0 else 'NONPOSITIVE_NONFAVORABLE',
                         'sign_changed_ols_to_theil_sen':bool((ov>0)!=(tv>0))})
    cases=[]; summary=[]
    for before,after,key,label in [('CMUR','DMC','CMUR_to_DMC','CMUR → DMC'),('DMC','RMC','DMC_to_RMC','DMC → RMC'),('TERR','CFOOT','TERR_to_CFOOT','Territorial → consumption-based GHG')]:
        persisted=0
        for geo in sorted(memberships[key]):
            p=(float(ts.loc[geo,before])>0)!=(float(ts.loc[geo,after])>0); persisted+=int(p)
            cases.append({'comparison':label,'comparison_key':key,'geo':geo,'ols_point_sign_disagreement':True,'theil_sen_disagreement_persists':bool(p),
                          'theil_sen_first_sign':'POSITIVE_FAVORABLE' if ts.loc[geo,before]>0 else 'NONPOSITIVE_NONFAVORABLE',
                          'theil_sen_second_sign':'POSITIVE_FAVORABLE' if ts.loc[geo,after]>0 else 'NONPOSITIVE_NONFAVORABLE'})
        summary.append({'comparison':label,'comparison_key':key,'ols_disagreement_n':len(memberships[key]),'theil_sen_persistence_n':persisted,
                        'theil_sen_persistence_fraction':persisted/len(memberships[key])})
    return pd.DataFrame(rows),pd.DataFrame(cases),pd.DataFrame(summary)


def trend_intervals(panel):
    rows=[]; full=progress_table(panel,EU27,YEARS); cc=consequences(full).set_index('geo'); members=set()
    for key in ['CMUR_to_DMC','DMC_to_RMC','TERR_to_CFOOT']: members.update(cc.index[cc[f'disagree_{key}']])
    implementation='statsmodels OLS with intercept and centered calendar year; conventional t interval; HC3; HAC/Newey-West lag 1/2 with use_correction=True and use_t=True'
    for geo in EU27:
        g=panel[panel.geo==geo].sort_values('year'); x=g.year.to_numpy(float); xc=x-x.mean(); X=sm.add_constant(xc)
        for rep,col in PRIMARY_COLS.items():
            y=transform_series(g[col].to_numpy(float),rep); fit=sm.OLS(y,X).fit(use_t=True)
            classic=fit.conf_int(alpha=.05)[1]; hc3=fit.get_robustcov_results(cov_type='HC3',use_t=True); hc3ci=hc3.conf_int(alpha=.05)[1]
            hac1=fit.get_robustcov_results(cov_type='HAC',maxlags=1,use_correction=True,use_t=True); h1=hac1.conf_int(alpha=.05)[1]
            hac2=fit.get_robustcov_results(cov_type='HAC',maxlags=2,use_correction=True,use_t=True); h2=hac2.conf_int(alpha=.05)[1]
            ex=lambda ci: bool(ci[0]>0 or ci[1]<0)
            rows.append({'geo':geo,'representation':rep,'n_years':14,'k_parameters':2,'df_resid':float(fit.df_resid),'slope':float(fit.params[1]),
                         'conventional_se':float(fit.bse[1]),'conventional_interval_low':float(classic[0]),'conventional_interval_high':float(classic[1]),'conventional_excludes_zero':ex(classic),
                         'hc3_se':float(hc3.bse[1]),'hc3_interval_low':float(hc3ci[0]),'hc3_interval_high':float(hc3ci[1]),'hc3_excludes_zero':ex(hc3ci),
                         'hac_lag1_se':float(hac1.bse[1]),'hac_lag1_interval_low':float(h1[0]),'hac_lag1_interval_high':float(h1[1]),'hac_lag1_excludes_zero':ex(h1),
                         'hac_lag2_se':float(hac2.bse[1]),'hac_lag2_interval_low':float(h2[0]),'hac_lag2_interval_high':float(h2[1]),'hac_lag2_excludes_zero':ex(h2),
                         'in_any_point_slope_sign_disagreement':geo in members,'implementation':implementation})
    d=pd.DataFrame(rows); cnt=[]
    for rep in REP_LABELS:
        q=d[d.representation==rep]
        for method,col in [('conventional','conventional_excludes_zero'),('HC3','hc3_excludes_zero'),('HAC_lag1','hac_lag1_excludes_zero'),('HAC_lag2','hac_lag2_excludes_zero')]:
            cnt.append({'representation':rep,'method':method,'interval_excludes_zero_n':int(q[col].sum()),'total_countries':len(q)})
    return d,pd.DataFrame(cnt)


def population_flag_rank_displacement_summary(panel):
    status=panel['population_status'].fillna('').astype(str).str.strip()
    flagged=sorted(panel.loc[status.ne(''),'geo'].unique().tolist())
    expected=['BE','CZ','DE','FR','HU','LU','LV','PL','PT','RO']
    if flagged!=expected:
        raise RuntimeError(f'population any-flag membership mismatch: {flagged} != {expected}')
    base=consequences(progress_table(panel,EU27,YEARS))
    defs=[
        ('CMUR_to_DMC','CMUR → DMC','shift_CMUR_to_DMC'),
        ('DMC_to_RMC','DMC → RMC','shift_DMC_to_RMC'),
        ('TERR_to_CFOOT','Territorial → consumption-based GHG','shift_TERR_to_CFOOT'),
    ]
    interpretation=('A country-level any-flag diagnostic did not show a uniform pattern of larger mean absolute rank displacement '
                    'among Member States carrying population-denominator status flags across all three comparison axes. '
                    'This diagnostic is descriptive and does not identify an effect of status flags.')
    rows=[]
    fset=set(flagged)
    for key,label,col in defs:
        for group,isflag in [('at_least_one_population_flag',True),('no_population_flag',False)]:
            q=base[base.geo.map(lambda g:(g in fset)==isflag)].copy()
            rows.append({'comparison_key':key,'comparison_label':label,'flag_group':group,'n_countries':int(len(q)),
                         'mean_abs_rank_displacement':float(q[col].abs().mean()),'countries':';'.join(sorted(q.geo.tolist())),
                         'interpretation':interpretation})
    return pd.DataFrame(rows)


def status_label(code):
    labels={'b':'break in time series','d':'definition differs (see metadata)','e':'estimated','f':'forecast','i':'value imputed by Eurostat or other receiving agencies','m':'missing value, data cannot exist','n':'not significant','p':'provisional','u':'low reliability'}
    if not code: return 'no observation-status flag'
    parts=[]
    for ch in code:
        parts.append(labels.get(ch,f'unmapped code {ch}'))
    return '; '.join(parts)


def source_status(panel):
    defs=[('CMUR','cmur_status'),('DMC','dmc_total_status'),('RMC','rmc_total_status'),('territorial_GHG','terr_total_status'),('consumption_GHG','cfoot_total_status'),('population_denominator','population_status')]
    rows=[]
    for source,col in defs:
        s=panel[col].fillna('').astype(str)
        for code in sorted(s.unique(),key=lambda x:(x!='',x)):
            q=panel[s==code]
            rows.append({'source':source,'status_column':col,'status_code':code if code else 'BLANK','official_status_label':status_label(code),'count':len(q),
                         'total_country_year_observations':len(panel),'countries_affected':';'.join(sorted(q.geo.unique())),'years_affected':';'.join(map(str,sorted(q.year.unique()))),
                         'official_code_list_source':'https://ec.europa.eu/eurostat/data/database?node_code=aact_eaa01','official_evidence_locator':'Database page → Flags → Observation status (Obs_status)'})
    return pd.DataFrame(rows)


def joint_block_bootstrap(panel, B=10000, block_lengths=(2,3), seed=20260812, chunk=500):
    series=[]; labels=[]
    for geo in EU27:
        g=panel[panel.geo==geo].sort_values('year')
        for rep,col in PRIMARY_COLS.items(): series.append(transform_series(g[col].to_numpy(float),rep)); labels.append((geo,rep))
    Y=np.column_stack(series); x=np.asarray(YEARS,float); xc=x-x.mean(); ss=float(np.dot(xc,xc)); beta=(xc[:,None]*Y).sum(axis=0)/ss
    intercept=Y.mean(axis=0)-beta*x.mean(); fitted=intercept[None,:]+x[:,None]*beta[None,:]; resid=Y-fitted; indices={(g,r):i for i,(g,r) in enumerate(labels)}
    rank_rows=[]; fav_rows=[]; dis_rows=[]; grid_rows=[]; endpoint_rows=[]; disp_rows=[]; extreme_rows=[]; paired_rows=[]
    observed=grid_record(grid(progress_table(panel,EU27,YEARS)))
    for L in block_lengths:
        rng=np.random.default_rng(seed+L); rank_collect={rep:[] for rep in REP_LABELS}; slope_sign_sum=np.zeros(len(labels),dtype=np.int64)
        dis_sum={key:np.zeros(27,dtype=np.int64) for key in ['CMUR_to_DMC','DMC_to_RMC','TERR_to_CFOOT']}; disp_collect={key:[] for key in dis_sum}; globals_=[]; done=0
        while done<B:
            n=min(chunk,B-done); nblocks=math.ceil(len(YEARS)/L); starts=rng.integers(0,len(YEARS),size=(n,nblocks)); offs=np.arange(L)
            idx=(starts[:,:,None]+offs[None,None,:])%len(YEARS); idx=idx.reshape(n,-1)[:,:len(YEARS)]
            ys=fitted[None,:,:]+resid[idx,:]; slopes=(ys*xc[None,:,None]).sum(axis=1)/ss; slope_sign_sum+=(slopes>0).sum(axis=0)
            rankm={}
            for rep in REP_LABELS:
                cols=[indices[(geo,rep)] for geo in EU27]; vals=np.round(slopes[:,cols],RANK_DECIMALS); rankm[rep]=rankdata(-vals,axis=1,method='average'); rank_collect[rep].append(rankm[rep])
            def rc(A,Bb):
                am=A-A.mean(axis=1,keepdims=True); bm=Bb-Bb.mean(axis=1,keepdims=True); return (am*bm).sum(axis=1)/np.sqrt((am*am).sum(axis=1)*(bm*bm).sum(axis=1))
            a=rc(rankm['CMUR'],rankm['TERR']); b=rc(rankm['DMC'],rankm['TERR']); c=rc(rankm['RMC'],rankm['TERR']); d=rc(rankm['CMUR'],rankm['CFOOT']); e=rc(rankm['DMC'],rankm['CFOOT']); f=rc(rankm['RMC'],rankm['CFOOT'])
            six=np.column_stack([a,b,c,d,e,f]); fullrange=six.max(axis=1)-six.min(axis=1); endpoint=f-a
            sdict={rep:slopes[:,[indices[(geo,rep)] for geo in EU27]]>0 for rep in REP_LABELS}; counts={}
            for before,after,key in [('CMUR','DMC','CMUR_to_DMC'),('DMC','RMC','DMC_to_RMC'),('TERR','CFOOT','TERR_to_CFOOT')]:
                dis=sdict[before]!=sdict[after]; dis_sum[key]+=dis.sum(axis=0); counts[key]=dis.sum(axis=1); disp_collect[key].append(rankm[after]-rankm[before])
            globals_.append(pd.DataFrame({'a_CMUR_TERR':a,'b_DMC_TERR':b,'c_RMC_TERR':c,'d_CMUR_CFOOT':d,'e_DMC_CFOOT':e,'f_RMC_CFOOT':f,'full_grid_range':fullrange,'selected_endpoint_difference':endpoint,
                                          'point_sign_disagreements_CMUR_to_DMC':counts['CMUR_to_DMC'],'point_sign_disagreements_DMC_to_RMC':counts['DMC_to_RMC'],'point_sign_disagreements_TERR_to_CFOOT':counts['TERR_to_CFOOT']}))
            done+=n
        glob=pd.concat(globals_,ignore_index=True)
        cell_matrix=glob[GRID_KEYS].to_numpy(float)
        min_idx=np.argmin(cell_matrix,axis=1); max_idx=np.argmax(cell_matrix,axis=1)
        for extreme,idx in [('minimum',min_idx),('maximum',max_idx)]:
            counts=np.bincount(idx,minlength=len(GRID_KEYS))
            for j,key in enumerate(GRID_KEYS):
                extreme_rows.append({'block_length':L,'extreme':extreme,'cell_key':key,'cell_label':GRID_LABELS[key],
                                     'count':int(counts[j]),'bootstrap_replicates':B,'frequency':float(counts[j]/B)})
        for col in ['a_CMUR_TERR','b_DMC_TERR','c_RMC_TERR','d_CMUR_CFOOT','e_DMC_CFOOT','f_RMC_CFOOT','full_grid_range']:
            v=glob[col].to_numpy(); grid_rows.append({'block_length':L,'quantity':col,'median':np.median(v),'diagnostic_q025':np.quantile(v,.025),'diagnostic_q975':np.quantile(v,.975),'mean':np.mean(v),'sd':np.std(v,ddof=1)})
        v=glob.selected_endpoint_difference.to_numpy(); endpoint_rows.append({'block_length':L,'median':np.median(v),'diagnostic_q025':np.quantile(v,.025),'diagnostic_q975':np.quantile(v,.975),'mean':np.mean(v),'sd':np.std(v,ddof=1),'positive_bootstrap_frequency':np.mean(v>0)})
        for contrast_id,contrast_label,upper_key,lower_key in CONTRAST_DEFS:
            v=(glob[upper_key]-glob[lower_key]).to_numpy(float)
            obs=float(observed[contrast_id])
            if obs>0: freq=float(np.mean(v>0))
            elif obs<0: freq=float(np.mean(v<0))
            else: freq=np.nan
            paired_rows.append({'contrast_id':contrast_id,'contrast_label':contrast_label,'block_length':int(L),
                                'observed_difference':obs,'bootstrap_median':float(np.median(v)),
                                'diagnostic_q025':float(np.quantile(v,.025)),'diagnostic_q975':float(np.quantile(v,.975)),
                                'frequency_with_observed_sign':freq,'B':int(B),
                                'interpretation':'Paired within-draw difference under the joint residual-block bootstrap; diagnostic ranges are not confidence intervals.'})
        for rep in REP_LABELS:
            R=np.vstack(rank_collect[rep])
            for j,geo in enumerate(EU27): rank_rows.append({'block_length':L,'representation':rep,'geo':geo,'rank_median':np.median(R[:,j]),'rank_diagnostic_q025':np.quantile(R[:,j],.025),'rank_diagnostic_q975':np.quantile(R[:,j],.975)})
        for i,(geo,rep) in enumerate(labels): fav_rows.append({'block_length':L,'geo':geo,'representation':rep,'favorable_slope_bootstrap_frequency':slope_sign_sum[i]/B})
        for key,arr in dis_sum.items():
            for j,geo in enumerate(EU27): dis_rows.append({'block_length':L,'comparison_key':key,'geo':geo,'point_slope_sign_disagreement_bootstrap_frequency':arr[j]/B})
        for key,parts in disp_collect.items():
            D=np.vstack(parts)
            for j,geo in enumerate(EU27): disp_rows.append({'block_length':L,'comparison_key':key,'geo':geo,'rank_displacement_median':np.median(D[:,j]),'rank_displacement_diagnostic_q025':np.quantile(D[:,j],.025),'rank_displacement_diagnostic_q975':np.quantile(D[:,j],.975),'mean_abs_rank_displacement_across_bootstrap':np.mean(np.abs(D[:,j]))})
    cfg={'method':'joint circular moving-block residual bootstrap','seed_base':seed,'stream_seed_rule':'seed_base + block_length','B_per_block_length':B,'block_lengths':list(block_lengths),'T':14,'series_vectors_per_year':135,
         'resampling_unit':'whole 135-dimensional year-residual vector; identical sampled block indices for every country and representation',
         'residual_model':'primary country-specific linear trend on favorable-oriented transformed series','rank_rule':'12-decimal numerical canonicalization then descending average ranks',
         'interpretation':'secondary bootstrap diagnostic; reported ranges are bootstrap diagnostic ranges and frequencies are bootstrap frequencies'}
    return cfg,pd.DataFrame(grid_rows),pd.DataFrame(endpoint_rows),pd.DataFrame(rank_rows),pd.DataFrame(fav_rows),pd.DataFrame(dis_rows),pd.DataFrame(disp_rows),pd.DataFrame(extreme_rows),pd.DataFrame(paired_rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--panel',type=Path,default=ROOT/'work/data/panel_2010_2023.csv'); ap.add_argument('--analysis',type=Path,default=ROOT/'work/analysis'); ap.add_argument('--bootstrap-b',type=int,default=10000); args=ap.parse_args()
    panel=pd.read_csv(args.panel,keep_default_na=False)
    if len(panel)!=378 or sorted(panel.geo.unique())!=sorted(EU27) or sorted(panel.year.astype(int).unique())!=YEARS: raise RuntimeError('analysis panel support mismatch')
    panel_diff,_=compare_panel(args.panel,ROOT/'data/panel_2010_2023.csv',1e-6)
    if panel_diff>1e-6: raise RuntimeError('analysis panel differs from the frozen reference panel')
    out=args.analysis/'supplementary_data'; out.mkdir(parents=True,exist_ok=True)
    symmetric_effects(panel).to_csv(out/'symmetric_grid_and_temporal_effects.csv',index=False)
    monotone_paths(panel).to_csv(out/'all_monotone_paths.csv',index=False)
    within_axis(panel).to_csv(out/'within_axis_concordance.csv',index=False)
    detail,cases,summary=loyo_sign_stability(panel); detail.to_csv(out/'country_point_sign_and_loyo_stability.csv',index=False); cases.to_csv(out/'point_sign_disagreement_loyo_cases.csv',index=False); summary.to_csv(out/'point_sign_disagreement_loyo_summary.csv',index=False)
    ts,cases_ts,sum_ts=theil_sen_signs(panel); ts.to_csv(out/'theil_sen_country_signs.csv',index=False); cases_ts.to_csv(out/'theil_sen_disagreement_cases.csv',index=False); sum_ts.to_csv(out/'theil_sen_disagreement_summary.csv',index=False)
    intervals,counts=trend_intervals(panel); intervals.to_csv(out/'trend_interval_diagnostics.csv',index=False); counts.to_csv(out/'trend_interval_counts.csv',index=False)
    source_status(panel).to_csv(out/'source_status_audit.csv',index=False)
    population_flag_rank_displacement_summary(panel).to_csv(out/'population_flag_rank_displacement_summary.csv',index=False)
    # Static, source-controlled official-method registries are copied into the analysis output; no live web scraping is required.
    for name in ['rmc_country_method_registry.csv','rmc_official_evidence_register.csv']:
        src=ROOT/'data/provenance'/name
        if not src.is_file(): raise RuntimeError(f'missing source-controlled provenance registry: {src}')
        pd.read_csv(src).to_csv(out/name,index=False)
    registry=pd.read_csv(ROOT/'data/provenance'/'rmc_country_method_registry.csv',keep_default_na=False)
    fixed,deletion,n20,groups=full_grid_authority(panel,registry)
    fixed.to_csv(out/'full_grid_fixed_sensitivities.csv',index=False)
    deletion.to_csv(out/'full_grid_deletion_runs.csv',index=False)
    n20.to_csv(out/'rmc_provenance_sensitivity_n20.csv',index=False)
    groups.to_csv(out/'rmc_provenance_group_grids.csv',index=False)
    cfg,gs,es,ri,sf,df,rd,extremes,paired=joint_block_bootstrap(panel,B=args.bootstrap_b)
    (out/'bootstrap_config.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    gs.to_csv(out/'bootstrap_grid_summary.csv',index=False); es.to_csv(out/'bootstrap_endpoint_summary.csv',index=False); ri.to_csv(out/'bootstrap_country_rank_diagnostic_intervals.csv',index=False)
    sf.to_csv(out/'bootstrap_slope_favorable_frequencies.csv',index=False); df.to_csv(out/'bootstrap_sign_disagreement_frequencies.csv',index=False); rd.to_csv(out/'bootstrap_rank_displacement_diagnostics.csv',index=False)
    extremes.to_csv(out/'bootstrap_grid_extreme_identity_frequencies.csv',index=False)
    paired.to_csv(out/'bootstrap_paired_contrast_summary.csv',index=False)
    bootstrap_full_grid_summary(gs,extremes).to_csv(out/'bootstrap_full_grid_robustness.csv',index=False)
    lineage={'source_panel_sha256':PANEL_SHA256,'producer_script':'code/05_build_extended_publication_evidence.py','producer_script_sha256':sha256_file(Path(__file__)),
             'trend_interval_methods':['conventional OLS','HC3','HAC lag 1','HAC lag 2'],
             'trend_interval_level':0.95,
             'bootstrap_config':cfg,
             'environment_note':'Exact package versions are controlled by requirements-lock.txt; this canonical method record excludes runtime-specific platform metadata.'}
    (out/'software_and_method.json').write_text(json.dumps(lineage,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(f'EXTENDED_PUBLICATION_EVIDENCE_PASS files={len(list(out.glob("*")))} bootstrap_B_per_block={args.bootstrap_b}')

if __name__=='__main__': main()
