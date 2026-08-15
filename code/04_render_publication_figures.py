#!/usr/bin/env python3
"""Build publication-facing source CSVs and artwork from validated analysis outputs.

This module is intentionally a publication layer around the validated scientific
pipeline. It does not estimate trends, ranks, correlations, contrasts, or
robustness statistics. It reads those validated objects and reorganizes them
into publication-facing tables/figures.
"""
from __future__ import annotations

from pathlib import Path
import argparse, ast, hashlib, math, shutil, tempfile, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import shapefile
from shapely.geometry import shape as shapely_shape, box, Point
from shapely.ops import transform as shp_transform
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
EU27 = ["AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","EL","HU","IE","IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE"]
PERIOD_ORDER = ["FULL_2010_2023","EARLY_2010_2016","LATE_2017_2023"]
PERIOD_LABEL = {"FULL_2010_2023":"2010–2023","EARLY_2010_2016":"2010–2016","LATE_2017_2023":"2017–2023"}
OKABE_BLUE = "#0072B2"
OKABE_ORANGE = "#E69F00"
DARK = "#333333"
GRID = "#E7E7E7"
W178 = 178/25.4
W85 = 85/25.4
DIVERGE = LinearSegmentedColormap.from_list("blue_white_orange", [OKABE_BLUE, "#F5F5F5", OKABE_ORANGE])
RANK_CMAP = plt.get_cmap("cividis_r")

matplotlib.rcParams.update({
    "font.family":"DejaVu Sans",
    "font.size":7.0,
    "axes.titlesize":7.5,
    "axes.labelsize":7.0,
    "xtick.labelsize":6.2,
    "ytick.labelsize":6.2,
    "legend.fontsize":6.0,
    "pdf.fonttype":42,
    "ps.fonttype":42,
    "svg.fonttype":"none",
    "svg.hashsalt":"eu27-material-climate-progress-publication",
})

FIGURE_FILES = {
    "figure_1_primary_concordance_grid":"figure_1_primary_concordance_grid",
    "figure_2_rank_concordance_geometry":"figure_2_rank_concordance_geometry",
    "figure_3_country_representation_consequences":"figure_3_country_representation_consequences",
    "figure_4_geography_progress_ranks":"figure_4_geography_progress_ranks",
    "figure_5_temporal_representation_sensitivity":"figure_5_temporal_representation_sensitivity",
    "figure_6_deletion_robustness":"figure_6_deletion_robustness",
    "figure_s1_temporal_concordance_matrices":"figure_s1_temporal_concordance_matrices",
    "figure_s2_geography_representation_shifts":"figure_s2_geography_representation_shifts",
    "figure_s3_geography_progress_scores":"figure_s3_geography_progress_scores",
    "figure_s4_progress_score_geometry":"figure_s4_progress_score_geometry",
    "figure_s5_broader_robustness_summary":"figure_s5_broader_robustness_summary",
    "figure_s6_sign_flip_overlap":"figure_s6_sign_flip_overlap",
    "figure_s7_country_discordance_matrix":"figure_s7_country_discordance_matrix",
}


def _read(analysis: Path, rel: str) -> pd.DataFrame:
    p = analysis / rel
    if not p.is_file():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def _primary_country(analysis: Path) -> pd.DataFrame:
    d = _read(analysis, "supplementary/country_results_all_specs.csv")
    q = d[(d["period"] == "FULL_2010_2023") & (d["analysis_label"] == "primary")].copy()
    if len(q) != 27 or sorted(q.geo.tolist()) != sorted(EU27):
        raise RuntimeError("Primary country-result set is not exactly EU27")
    return q.sort_values("geo").reset_index(drop=True)


def _primary_grid_row(analysis: Path, correlation: str = "spearman") -> pd.Series:
    d = _read(analysis, "tables/representation_grid.csv")
    q = d[(d.period == "FULL_2010_2023") & (d.spec == "primary_common_pc") & (d.estimator == "OLS") & (d.cmur_mode == "pp") & (d.correlation == correlation)]
    if len(q) != 1:
        raise RuntimeError(f"Expected one primary {correlation} grid row")
    return q.iloc[0]


def _cell_rows_from_wide(df: pd.DataFrame) -> pd.DataFrame:
    mapping = [
        ("a_CMUR_TERR","CMUR","Territorial GHG"),
        ("b_DMC_TERR","DMC","Territorial GHG"),
        ("c_RMC_TERR","RMC","Territorial GHG"),
        ("d_CMUR_CFOOT","CMUR","Consumption-based GHG footprint"),
        ("e_DMC_CFOOT","DMC","Consumption-based GHG footprint"),
        ("f_RMC_CFOOT","RMC","Consumption-based GHG footprint"),
    ]
    rows=[]
    for _,r in df.iterrows():
        for col,mat,clim in mapping:
            rows.append({"period":r["period"],"material_representation":mat,"climate_perspective":clim,"spearman_rho":float(r[col])})
    return pd.DataFrame(rows)


def build_publication_sources(analysis: Path, output_root: Path) -> None:
    """Create publication source files from validated analysis and extended evidence outputs."""
    figsd=output_root/'figures'/'source_data'; tables=output_root/'results'/'reference'/'tables'; supp=output_root/'supplementary_tables'; suppdata=output_root/'supplementary_data'
    for d in [figsd,tables,supp,suppdata]: d.mkdir(parents=True,exist_ok=True)
    # Supplement table numbering is canonical S1–S9 in this publication layer; remove stale prior-numbered CSVs from the target.
    for p in supp.glob('table_s*.csv'):
        p.unlink()

    main=_read(analysis,'tables/main_results.csv')
    country=_primary_country(analysis)
    rg=_read(analysis,'tables/representation_grid.csv')
    rank_summary=_read(analysis,'tables/country_rank_change_summary.csv')
    robustness=_read(analysis,'supplementary/robustness_summary.csv')
    loco=_read(analysis,'supplementary/leave_one_country_out.csv')
    loyo=_read(analysis,'supplementary/leave_one_year_out.csv')
    cmur_even=_read(analysis,'supplementary/cmur_even_year.csv')
    ext=analysis/'supplementary_data'
    if not ext.is_dir(): raise RuntimeError('Extended publication evidence is missing; run code/05_build_extended_publication_evidence.py first')
    sym=pd.read_csv(ext/'symmetric_grid_and_temporal_effects.csv')
    paths=pd.read_csv(ext/'all_monotone_paths.csv')
    within=pd.read_csv(ext/'within_axis_concordance.csv')
    loyo_summary=pd.read_csv(ext/'point_sign_disagreement_loyo_summary.csv')
    source_status=pd.read_csv(ext/'source_status_audit.csv')
    trend_interval_counts=pd.read_csv(ext/'trend_interval_counts.csv')
    rmc_registry=pd.read_csv(ext/'rmc_country_method_registry.csv')
    full_grid_fixed=pd.read_csv(ext/'full_grid_fixed_sensitivities.csv')
    full_grid_deletion=pd.read_csv(ext/'full_grid_deletion_runs.csv')
    bootstrap_full_grid=pd.read_csv(ext/'bootstrap_full_grid_robustness.csv')
    bootstrap_grid_summary=pd.read_csv(ext/'bootstrap_grid_summary.csv')
    bootstrap_endpoint_summary=pd.read_csv(ext/'bootstrap_endpoint_summary.csv')
    bootstrap_paired=pd.read_csv(ext/'bootstrap_paired_contrast_summary.csv')
    population_flag_summary=pd.read_csv(ext/'population_flag_rank_displacement_summary.csv')
    rmc_n20=pd.read_csv(ext/'rmc_provenance_sensitivity_n20.csv')
    rmc_groups=pd.read_csv(ext/'rmc_provenance_group_grids.csv')
    rmc_status=_read(analysis,'supplementary/rmc_status.csv')

    # Copy all accepted machine-readable supplementary evidence into the publication layer.
    for p in sorted(ext.iterdir()):
        if p.is_file(): shutil.copy2(p,suppdata/p.name)

    # MAIN Table 2: explicit indicator/accounting taxonomy and provenance boundaries.
    t2=pd.DataFrame([
      ['CMUR','Material indicator: circularity share','Material-indicator / construct choice','Percent','OLS slope in percentage points/year','env_ac_cur','Official Eurostat indicator; annual 2010–2023'],
      ['DMC','Direct material throughput','Material-indicator / construct choice; domestic side of material accounting-boundary comparison','Tonnes per capita','Negative OLS slope of log per-capita level','env_ac_mfa','Official economy-wide material-flow account; common average-population denominator in primary analysis'],
      ['RMC','Final-demand material footprint','Material accounting-boundary choice','Tonnes per capita','Negative OLS slope of log per-capita level','env_ac_rme','Official model-based environmental-account series; country production routes can differ and cross-country comparability requires qualification'],
      ['Territorial GHG','Territorial greenhouse-gas inventory measure','GHG accounting-perspective choice','Tonnes CO2e per capita','Negative OLS slope of log per-capita level','sdg_13_10','Official territorial series; common average-population denominator in primary analysis'],
      ['Consumption-based GHG footprint','Final-demand greenhouse-gas footprint','GHG accounting-perspective choice','Tonnes CO2e per capita','Negative OLS slope of log per-capita level','cli_gge_foot','Model-assisted Eurostat footprint series using a common modelling method across countries; complete series is re-estimated with releases'],
    ],columns=['representation','scientific_role','construct_or_accounting_choice_category','unit','primary_progress_scale','official_source','model_or_provenance_note'])
    t2.to_csv(tables/'table_2_representation_contract_source.csv',index=False)

    # MAIN Table 3: symmetric Full-grid system; the selected endpoint is explicitly secondary.
    full=sym[sym.period=='FULL_2010_2023'].copy(); rows=[]
    order=['grid_cell','full_grid_minimum','full_grid_maximum','full_grid_range','material_simple_difference','climate_simple_difference','selected_endpoint_cell_difference']
    for typ in order:
        for _,rr in full[full.object_type==typ].iterrows():
            note='Primary symmetric system-level evidence'
            if typ=='selected_endpoint_cell_difference': note='Secondary descriptive difference between two selected grid cells'
            elif typ in ('material_simple_difference','climate_simple_difference'): note='Descriptive conditional difference; not a causal effect'
            rows.append(['A_full_grid',typ,rr.effect,float(rr.value),'Spearman rho' if typ in ('grid_cell','full_grid_minimum','full_grid_maximum') else 'Difference in Spearman rho',27,note])
    for _,rr in within.iterrows():
        rows.append(['B_within_axis','within_axis_concordance',rr.comparison,float(rr.spearman),'Spearman rho',27,'Calibration of similarity between country progress orderings within an axis'])
    pd.DataFrame(rows,columns=['panel','object_type','public_label','value','metric','n_countries','interpretation_note']).to_csv(tables/'table_3_primary_concordance_and_contrasts.csv',index=False)

    # MAIN Table 4: existing rank arithmetic plus point-slope sign and LOYO stability.
    labelmap={'CMUR_to_DMC':'CMUR → DMC','DMC_to_RMC':'DMC → RMC','TERR_to_CFOOT':'Territorial → consumption-based GHG'}
    t4=rank_summary.rename(columns={'substitution':'comparison_key','max_country':'country_at_maximum','sign_flips_n':'point_slope_sign_disagreements_n'}).copy()
    t4.insert(1,'comparison',t4.comparison_key.map(labelmap))
    stable=loyo_summary[['comparison_key','both_sign_loyo_stable_n','disagreement_denominator_n','both_sign_loyo_stable_fraction']]
    t4=t4.merge(stable,on='comparison_key',how='left',validate='one_to_one')
    t4.to_csv(tables/'table_4_country_consequences.csv',index=False)

    # MAIN Table 5: robustness/sensitivity of the primary six-cell grid.
    panel_heading='Panel A — Grid-level robustness and sensitivity'
    t5=[]
    fixed_metadata={
        'baseline':('full_reference','Full 2010–2023 baseline','n = 27','Point summary of the six-cell grid; the range is a descriptive spread across the six cells.'),
        'common_log_CMUR':('scale_or_estimand_convention','Common-log CMUR','n = 27','Point summary of the six-cell grid; the range is a descriptive spread across the six cells.'),
        'Theil_Sen':('estimator_sensitivity','Theil–Sen','n = 27','Point summary of the six-cell grid; the range is a descriptive spread across the six cells.'),
        'total_scale':('scale_or_estimand_convention','Total scale','n = 27','Point summary of the six-cell grid; the range is a descriptive spread across the six cells.'),
        'common_even_year':('temporal_support_convention','Common even-year','n = 27','Point summary of the six-cell grid; the range is a descriptive spread across the six cells.'),
        'exclude_2020_2021':('event_defined_temporal_sensitivity','Exclude 2020 and 2021','n = 27; 12 retained calendar years','All five trajectories re-estimated after excluding calendar years 2020 and 2021.'),
    }
    for key,(klass,label,scope,interpretation) in fixed_metadata.items():
        q=full_grid_fixed[full_grid_fixed.diagnostic_key.eq(key)]
        if len(q)!=1: raise RuntimeError(f'Missing full-grid fixed sensitivity: {key}')
        rr=q.iloc[0]
        t5.append(['A_grid_level',panel_heading,klass,key,label,scope,'Full-grid range','Difference in Spearman rho',float(rr.full_grid_range),float(rr.full_grid_range),float(rr.full_grid_range),
                   rr.full_grid_minimum_cell_label,rr.full_grid_maximum_cell_label,interpretation])
    for dtype,label in [('LOCO','Leave-one-country-out'),('LOYO','Leave-one-year-out')]:
        q=full_grid_deletion[full_grid_deletion.deletion_type.eq(dtype)]
        if q.empty: raise RuntimeError(f'Missing {dtype} full-grid deletion runs')
        ref=float(full_grid_deletion.loc[full_grid_deletion.deletion_type.eq('FULL'),'full_grid_range'].iloc[0])
        scope='27 runs; n = 26 each' if dtype=='LOCO' else '14 runs; n = 27 each'
        t5.append(['A_grid_level',panel_heading,'deletion_robustness',dtype,label,scope,'Full-grid range','Difference in Spearman rho',ref,float(q.full_grid_range.min()),float(q.full_grid_range.max()),'', '',
                   'Deletion-run interval across full-grid ranges; the reference is the Full 2010–2023 grid range.'])
    for _,rr in bootstrap_full_grid.sort_values('block_length').iterrows():
        L=int(rr.block_length)
        t5.append(['A_grid_level',panel_heading,'bootstrap_diagnostic',f'bootstrap_block_{L}',f'Joint residual-block bootstrap, block {L}','n = 27; B = 10,000','Full-grid range','Difference in Spearman rho',
                   float(rr.full_grid_range_median),float(rr.full_grid_range_diagnostic_q025),float(rr.full_grid_range_diagnostic_q975),
                   rr.modal_minimum_cell_label,rr.modal_maximum_cell_label,'Median and 2.5–97.5% bootstrap diagnostic range; cell identities shown are modal extreme-cell identities across realizations.'])
    if len(rmc_n20)!=1: raise RuntimeError('Expected one n=20 RMC provenance-sensitivity row')
    rr=rmc_n20.iloc[0]
    t5.append(['A_grid_level',panel_heading,'provenance_composition_sensitivity','rmc_exclude_mixed_changed','Additional RMC-provenance sensitivity excluding countries classified as mixed/changed in the RMC provenance registry','n = 20','Full-grid range','Difference in Spearman rho',
               float(rr.full_grid_range),float(rr.full_grid_range),float(rr.full_grid_range),rr.full_grid_minimum_cell_label,rr.full_grid_maximum_cell_label,
               'Common n = 20 after excluding countries classified as mixed/changed in the RMC provenance registry; provenance regime and country composition are confounded.'])
    pd.DataFrame(t5,columns=['panel','panel_heading','evidence_class','diagnostic_key','public_label','scope_or_n','analytical_object','metric','reference_or_point','lower_value','upper_value','minimum_cell_context','maximum_cell_context','interpretation']).to_csv(tables/'table_5_temporal_and_robustness.csv',index=False)

    # Supplement Table S1: source registry retained with compact provenance note.
    sources=pd.read_csv(ROOT/'data'/'sources.csv')
    s1=sources[['source_id','dataset','role','raw_file','api_query','resolved_selectors','rows','countries','years']].copy()
    def _selector_label(raw,dimension):
        try:
            items=ast.literal_eval(raw) if isinstance(raw,str) else []
            for item in items:
                if item.get('dimension')==dimension: return item.get('label') or item.get('code') or ''
        except Exception: return ''
        return ''
    s1['unit']=s1.resolved_selectors.map(lambda x:_selector_label(x,'unit'))
    s1.loc[s1.source_id.eq('average_population_demo_gind')&s1.unit.eq(''),'unit']='Persons (average population)'
    s1['indicator_or_measure']=s1['role']; s1['provenance_note']='Frozen Eurostat response in data/raw; exact query and resolved selectors recorded in data/sources.csv'
    s1=s1[['source_id','dataset','indicator_or_measure','unit','role','raw_file','api_query','rows','countries','years','provenance_note']]
    s1.to_csv(supp/'table_s1_sources_units_provenance.csv',index=False)

    # S2 and S3 remain the detailed temporal grid and companion correlations.
    s2=_cell_rows_from_wide(main); s2['period_label']=s2.period.map(PERIOD_LABEL); s2.to_csv(supp/'table_s2_temporal_concordance_matrices.csv',index=False)
    comp=[]; base_rg=rg[(rg.spec=='primary_common_pc')&(rg.estimator=='OLS')&(rg.cmur_mode=='pp')]
    for period in PERIOD_ORDER:
        sr=base_rg[(base_rg.period==period)&(base_rg.correlation=='spearman')].iloc[0]
        for cr in ['kendall','pearson']:
            rr=base_rg[(base_rg.period==period)&(base_rg.correlation==cr)].iloc[0]
            for key,mat,clim in [('a_CMUR_TERR','CMUR','Territorial GHG'),('b_DMC_TERR','DMC','Territorial GHG'),('c_RMC_TERR','RMC','Territorial GHG'),('d_CMUR_CFOOT','CMUR','Consumption-based GHG footprint'),('e_DMC_CFOOT','DMC','Consumption-based GHG footprint'),('f_RMC_CFOOT','RMC','Consumption-based GHG footprint')]:
                comp.append([period,mat,clim,cr,float(rr[key]),float(sr[key]),27])
    pd.DataFrame(comp,columns=['period','material_representation','climate_perspective','correlation','correlation_value','spearman_reference','n_countries']).to_csv(supp/'table_s3_companion_correlations.csv',index=False)

    # S4: 9 monotone-path rows plus 11 symmetric rows for each of Full, Early and Late.
    s4=[]
    for _,rr in paths.iterrows():
        s4.append(['monotone_path','FULL_2010_2023',rr.path,int(rr.step_number),rr.step_sequence,float(rr.step_value),float(rr.path_sum),float(rr.selected_endpoint_difference),'Path allocation is descriptive; all allowed paths share the same selected endpoint sum'])
    symmetric_types=['full_grid_range','material_simple_difference','climate_simple_difference','selected_endpoint_cell_difference']
    for period in ['FULL_2010_2023','EARLY_2010_2016','LATE_2017_2023']:
        q=sym[(sym.period==period)&sym.object_type.isin(symmetric_types)]
        if len(q)!=11: raise RuntimeError(f'Expected 11 symmetric S4 rows for {period}, found {len(q)}')
        for _,rr in q.iterrows():
            interp='Descriptive Full-period symmetric grid summary' if period=='FULL_2010_2023' else 'Descriptive temporal heterogeneity; no structural-break interpretation'
            s4.append(['symmetric_temporal',period,'',0,rr.effect,float(rr.value),np.nan,np.nan,interp])
    s4df=pd.DataFrame(s4,columns=['record_type','period','path','step_number','public_label','value','path_sum','selected_endpoint_difference','interpretation'])
    if len(s4df)!=42: raise RuntimeError(f'Table S4 row-count mismatch: {len(s4df)} != 42')
    s4df.to_csv(supp/'table_s4_temporal_ordered_contrasts.csv',index=False)

    # S5: Panel A mirrors MAIN Table 5, Panel B keeps 10 selected-endpoint diagnostics, Panel C has 20 interval rows.
    s5_cols=['panel','panel_heading','evidence_class','diagnostic_key','public_label','scope_or_n','analytical_object','metric','reference_or_point','lower_value','upper_value','minimum_cell_context','maximum_cell_context','representation','method','interval_excludes_zero_n','total_countries','interpretation']
    s5=[]
    for row in t5:
        panel,phead,klass,key,label,scope,obj,metric,point,low,high,minctx,maxctx,interp=row
        s5.append([panel,phead,klass,key,label,scope,obj,metric,point,low,high,minctx,maxctx,'','',np.nan,np.nan,interp])

    full_rg=rg[(rg.period=='FULL_2010_2023')&(rg.correlation=='spearman')]
    endpoint_rows=[]
    endpoint_order=[
        ('primary_common_pc','OLS','pp','full_reference','Full 2010–2023 reference','n = 27','Full-period selected endpoint; secondary descriptive evidence'),
        ('published_pc','OLS','pp','published_pc','Published per-capita series','n = 27','Alternative denominator convention'),
        ('primary_common_pc','OLS','log','common_log_CMUR','Common-log CMUR sensitivity','n = 27','Alternative CMUR progress-scale convention'),
        ('primary_common_pc','THEIL_SEN','pp','Theil_Sen','Theil–Sen','n = 27','Alternative country-trend estimator'),
        ('total_scale','OLS','pp','total_scale','Total-scale sensitivity','n = 27','Alternative scale/estimand convention'),
    ]
    for spec,est,mode,key,label,scope,interp in endpoint_order:
        q=full_rg[(full_rg.spec==spec)&(full_rg.estimator==est)&(full_rg.cmur_mode==mode)]
        if len(q)!=1: raise RuntimeError(f'Missing endpoint fixed diagnostic {key}')
        val=float(q.iloc[0].D_TOTAL_f_minus_a)
        endpoint_rows.append(['B_selected_endpoint','Panel B — Selected-endpoint sensitivity (secondary descriptive evidence)','secondary_endpoint',key,label,scope,'Selected endpoint cell difference','Difference in Spearman rho',val,val,val,'','','','',np.nan,np.nan,interp])
    for dtype,label in [('LOCO','Leave-one-country-out'),('LOYO','Leave-one-year-out')]:
        q=robustness[(robustness.estimand=='D_TOTAL_f_minus_a')&(robustness.diagnostic==dtype)]
        if len(q)!=1: raise RuntimeError(f'Missing selected-endpoint {dtype} summary')
        rr=q.iloc[0]
        scope='27 runs; n = 26 each' if dtype=='LOCO' else '14 runs; n = 27 each'
        endpoint_rows.append(['B_selected_endpoint','Panel B — Selected-endpoint sensitivity (secondary descriptive evidence)','secondary_endpoint_deletion',dtype,label,scope,'Selected endpoint cell difference','Difference in Spearman rho',np.nan,float(rr['min']),float(rr['max']),'','','','',np.nan,np.nan,'Deletion perturbation; selected endpoint definition unchanged'])
    ce=cmur_even.iloc[0]
    endpoint_rows.append(['B_selected_endpoint','Panel B — Selected-endpoint sensitivity (secondary descriptive evidence)','secondary_endpoint_temporal_support','CMUR_even_year','Common even-year sensitivity','n = 27; 7 retained calendar years','Selected endpoint cell difference','Difference in Spearman rho',float(ce.D_TOTAL_f_minus_a),float(ce.D_TOTAL_f_minus_a),float(ce.D_TOTAL_f_minus_a),'','','','',np.nan,np.nan,'Common even-year temporal-support convention'])
    for _,rr in bootstrap_endpoint_summary.sort_values('block_length').iterrows():
        L=int(rr.block_length)
        endpoint_rows.append(['B_selected_endpoint','Panel B — Selected-endpoint sensitivity (secondary descriptive evidence)','secondary_endpoint_bootstrap',f'bootstrap_block_{L}',f'Joint residual-block bootstrap, block {L}','n = 27; B = 10,000','Selected endpoint cell difference','Difference in Spearman rho',float(rr['median']),float(rr.diagnostic_q025),float(rr.diagnostic_q975),'','','','',np.nan,np.nan,'Model-dependent joint residual-block bootstrap diagnostic; selected endpoint remains secondary'])
    if len(endpoint_rows)!=10: raise RuntimeError(f'Table S5 Panel B row-count mismatch: {len(endpoint_rows)} != 10')
    s5.extend(endpoint_rows)
    for _,rr in trend_interval_counts.iterrows():
        s5.append(['C_trend_intervals','Panel C — Country trend-interval summary','trend_interval_summary',f"{rr.representation}|{rr.method}",f"{rr.representation} — {rr.method}",'n = 27','Country trend interval excludes-zero count','Countries (of 27)',np.nan,np.nan,np.nan,'','',rr.representation,rr.method,int(rr.interval_excludes_zero_n),int(rr.total_countries),'Compact interval summary only; not a significant/non-significant country classification'])
    s5df=pd.DataFrame(s5,columns=s5_cols)
    if len(s5df)!=41 or (s5df.panel=='A_grid_level').sum()!=11 or (s5df.panel=='B_selected_endpoint').sum()!=10 or (s5df.panel=='C_trend_intervals').sum()!=20:
        raise RuntimeError('Table S5 row-count/panel mismatch')
    s5df.to_csv(supp/'table_s5_extended_robustness.csv',index=False)

    # S6: reader-facing cellwise and paired within-draw temporal-bootstrap diagnostics.
    scope_note=('These diagnostics perturb the temporal residual structure conditional on the disseminated series and fitted trend specification; '
                'they do not propagate uncertainty from the upstream statistical-production models used to construct model-assisted footprint series.')
    baseline_fixed=full_grid_fixed[full_grid_fixed.diagnostic_key.eq('baseline')]
    if len(baseline_fixed)!=1: raise RuntimeError('Missing baseline fixed grid for Table S6')
    br=baseline_fixed.iloc[0]
    grid_labels={'a_CMUR_TERR':'CMUR × territorial','b_DMC_TERR':'DMC × territorial','c_RMC_TERR':'RMC × territorial','d_CMUR_CFOOT':'CMUR × consumption','e_DMC_CFOOT':'DMC × consumption','f_RMC_CFOOT':'RMC × consumption'}
    s6=[]
    for L in [2,3]:
        q=bootstrap_grid_summary[bootstrap_grid_summary.block_length.eq(L)].set_index('quantity')
        for key in grid_labels:
            rr=q.loc[key]
            s6.append(['A_cellwise',key,grid_labels[key],L,float(br[key]),float(rr['median']),float(rr.diagnostic_q025),float(rr.diagnostic_q975),np.nan,'Spearman rho',scope_note])
    for _,rr in bootstrap_paired.sort_values(['block_length','contrast_id']).iterrows():
        s6.append(['B_paired_contrasts',rr.contrast_id,rr.contrast_label,int(rr.block_length),float(rr.observed_difference),float(rr.bootstrap_median),float(rr.diagnostic_q025),float(rr.diagnostic_q975),float(rr.frequency_with_observed_sign),'Difference in Spearman rho',scope_note+' Paired contrasts use the two cells from the same bootstrap draw.'])
    s6df=pd.DataFrame(s6,columns=['panel','object_id','object_label','block_length','observed_value','bootstrap_median','diagnostic_q025','diagnostic_q975','frequency_with_observed_sign','metric','interpretation'])
    if len(s6df)!=30 or (s6df.panel=='A_cellwise').sum()!=12 or (s6df.panel=='B_paired_contrasts').sum()!=18:
        raise RuntimeError('Table S6 row-count/panel mismatch')
    s6df.to_csv(supp/'table_s6_temporal_bootstrap_diagnostics.csv',index=False)

    # S7: production/residence accounting bridge, wide architecture retained.
    s7=_read(analysis,'supplementary/accounting_bridge.csv')
    if len(s7)!=9: raise RuntimeError(f'Table S7 row-count mismatch: {len(s7)} != 9')
    s7.to_csv(supp/'table_s7_production_account_bridge.csv',index=False)

    # S8: RMC provenance registry (27), common n=20 material rows (3), and three-group material rows (9).
    s8=[]
    s8_cols=['panel','record_type','geo','provenance_group','material_measure','territorial_ghg','consumption_ghg','full_grid_range','n_countries','countries','disseminated_source_regime','documented_estimation_method','documented_coverage','official_evidence_source','official_evidence_locator','confidence','interpretation']
    for _,rr in rmc_registry.iterrows():
        s8.append(['A_country_registry','country_registry',rr.geo,'','',np.nan,np.nan,np.nan,1,rr.geo,rr.disseminated_source_regime,rr.documented_estimation_method,rr.documented_coverage,rr.official_evidence_source,rr.official_evidence_locator,rr.confidence,'Country-level registry entry; exact year-level production route is not inferred beyond official documentation.'])
    n20=rmc_n20.iloc[0]
    for mat,tkey,ckey in [('CMUR','a_CMUR_TERR','d_CMUR_CFOOT'),('DMC','b_DMC_TERR','e_DMC_CFOOT'),('RMC','c_RMC_TERR','f_RMC_CFOOT')]:
        s8.append(['B_rmc_provenance_sensitivity','n20_sensitivity','','',mat,float(n20[tkey]),float(n20[ckey]),float(n20.full_grid_range),int(n20.n_countries),n20.included_countries,'','','','','','',n20.composition_caveat])
    for _,rr in rmc_groups.iterrows():
        for mat,tkey,ckey in [('CMUR','a_CMUR_TERR','d_CMUR_CFOOT'),('DMC','b_DMC_TERR','e_DMC_CFOOT'),('RMC','c_RMC_TERR','f_RMC_CFOOT')]:
            s8.append(['C_registry_groups','provenance_group_grid','',rr.provenance_group,mat,float(rr[tkey]),float(rr[ckey]),float(rr.full_grid_range),int(rr.n_countries),rr.countries,rr.registry_regime,'','','','','',rr.composition_caveat])
    s8df=pd.DataFrame(s8,columns=s8_cols)
    if len(s8df)!=39 or (s8df.panel=='A_country_registry').sum()!=27 or (s8df.panel=='B_rmc_provenance_sensitivity').sum()!=3 or (s8df.panel=='C_registry_groups').sum()!=9:
        raise RuntimeError('Table S8 row-count/panel mismatch')
    s8df.to_csv(supp/'table_s8_rmc_provenance.csv',index=False)

    # S9: source status, denominator sensitivity, direct tie rows, two selected scale diagnostics, and population any-flag diagnostic.
    s9=[]
    s9_cols=['panel','object_id','object_label','group_or_context','n_countries','metric','value','auxiliary_value_1','auxiliary_value_2','countries','note']
    for _,rr in source_status.iterrows():
        s9.append(['A_source_status',f"{rr.source}|{rr.status_code}",rr.source,rr.status_code,np.nan,'Country-year observations',float(rr['count']),'','',rr.countries_affected,rr.official_status_label])
    pcs=_read(analysis,'supplementary/published_vs_common_rank_summary.csv')
    for _,rr in pcs.iterrows():
        s9.append(['B_published_vs_common',f"{rr.period}|{rr.representation}",rr.representation,rr.period,27,'Spearman rho',float(rr.spearman_progress_common_vs_published),float(rr.mean_abs_rank_displacement),float(rr.max_abs_rank_displacement),'','Auxiliary values are mean and maximum absolute rank displacement.'])
    ties=_read(analysis,'supplementary/tie_audit.csv')
    for i,rr in ties.reset_index(drop=True).iterrows():
        ctx=f"{rr.period}|{rr.spec}|{rr.estimator}|{rr.cmur_mode}|{rr.variable}"
        s9.append(['C_tie_groups',f'tie_{i+1:02d}',rr.variable,ctx,int(rr.n_tied),'Canonical tied value',float(rr.canonical_value_12dp),float(rr.raw_span),'',rr.geos,'12-decimal numerical canonicalization prevents implementation-dependent rank splitting of machine-scale numerical ties.'])
    for key,label in [('common_even_year','Common even-year sensitivity'),('total_scale','Total-scale sensitivity')]:
        q=full_grid_fixed[full_grid_fixed.diagnostic_key.eq(key)]
        if len(q)!=1: raise RuntimeError(f'Missing S9 selected scale diagnostic {key}')
        rr=q.iloc[0]
        s9.append(['D_selected_scale',key,label,'selected endpoint (secondary)',27,'Difference in Spearman rho',float(rr.selected_endpoint_cell_difference),'','', '', 'Selected endpoint is secondary descriptive evidence; full-grid evidence is reported separately.'])
    for _,rr in population_flag_summary.iterrows():
        s9.append(['E_population_any_flag',f"{rr.comparison_key}|{rr.flag_group}",rr.comparison_label,rr.flag_group,int(rr.n_countries),'Mean absolute rank displacement',float(rr.mean_abs_rank_displacement),'','',rr.countries,rr.interpretation])
    s9df=pd.DataFrame(s9,columns=s9_cols)
    if len(s9df)!=46 or (s9df.panel=='A_source_status').sum()!=14 or (s9df.panel=='B_published_vs_common').sum()!=12 or (s9df.panel=='C_tie_groups').sum()!=12 or (s9df.panel=='D_selected_scale').sum()!=2 or (s9df.panel=='E_population_any_flag').sum()!=6:
        raise RuntimeError('Table S9 row-count/panel mismatch')
    s9df.to_csv(supp/'table_s9_source_denominator_tie_scale_diagnostics.csv',index=False)

    # Figure 1: symmetric Full 3×2 grid; values unchanged.
    r=main[main.period=='FULL_2010_2023'].iloc[0]; f1=[]
    for key,mat,clim in [('a_CMUR_TERR','CMUR','Territorial GHG'),('b_DMC_TERR','DMC','Territorial GHG'),('c_RMC_TERR','RMC','Territorial GHG'),('d_CMUR_CFOOT','CMUR','Consumption-based GHG footprint'),('e_DMC_CFOOT','DMC','Consumption-based GHG footprint'),('f_RMC_CFOOT','RMC','Consumption-based GHG footprint')]: f1.append([mat,clim,float(r[key])])
    pd.DataFrame(f1,columns=['material_representation','climate_perspective','spearman_rho']).to_csv(figsd/'figure_1_primary_concordance_grid.csv',index=False)

    # Figure 2 unchanged.
    rr=[]
    for mat in ['CMUR','DMC','RMC']:
        for clim,clabel in [('TERR','Territorial GHG'),('CFOOT','Consumption-based GHG footprint')]:
            key={'CMUR_TERR':'a_CMUR_TERR','DMC_TERR':'b_DMC_TERR','RMC_TERR':'c_RMC_TERR','CMUR_CFOOT':'d_CMUR_CFOOT','DMC_CFOOT':'e_DMC_CFOOT','RMC_CFOOT':'f_RMC_CFOOT'}[f'{mat}_{clim}']
            rho=float(r[key])
            for _,cc in country.iterrows(): rr.append([f'{mat}_{clim}',f'{mat} × {clabel}',cc.geo,float(cc[f'rank_{mat}']),float(cc[f'rank_{clim}']),rho])
    pd.DataFrame(rr,columns=['panel','panel_label','geo','material_rank','climate_rank','spearman_rho']).to_csv(figsd/'figure_2_rank_concordance_geometry.csv',index=False)

    # Figure 3/S2 point-estimate source; terminology is explicit while the arithmetic is unchanged.
    cc=country[['geo','rank_shift_CMUR_to_DMC','rank_shift_DMC_to_RMC','rank_shift_TERR_to_CFOOT','sign_flip_CMUR_DMC','sign_flip_DMC_RMC','sign_flip_TERR_CFOOT']].copy()
    cc=cc.rename(columns={'sign_flip_CMUR_DMC':'point_sign_disagreement_CMUR_DMC','sign_flip_DMC_RMC':'point_sign_disagreement_DMC_RMC','sign_flip_TERR_CFOOT':'point_sign_disagreement_TERR_CFOOT'})
    cc.to_csv(figsd/'figure_3_country_representation_consequences.csv',index=False)
    country[['geo','rank_CMUR','rank_DMC','rank_RMC','rank_TERR','rank_CFOOT']].to_csv(figsd/'figure_4_geography_progress_ranks.csv',index=False)
    main[['role','period','a_CMUR_TERR','b_DMC_TERR','c_RMC_TERR','d_CMUR_CFOOT','e_DMC_CFOOT','f_RMC_CFOOT']].to_csv(figsd/'figure_5_temporal_representation_sensitivity.csv',index=False)

    # Figure 6: full-grid-range deletion robustness only. Source retains the six cells and extreme-cell identities, with no endpoint series.
    f6=full_grid_deletion.rename(columns={'deletion_type':'source','deletion_id':'omitted_unit'}).copy()
    keep=['source','omitted_unit','n_countries','n_years','cell_metric','full_grid_range_metric',
          'a_CMUR_TERR','b_DMC_TERR','c_RMC_TERR','d_CMUR_CFOOT','e_DMC_CFOOT','f_RMC_CFOOT',
          'full_grid_minimum','full_grid_minimum_cell_key','full_grid_minimum_cell_label',
          'full_grid_maximum','full_grid_maximum_cell_key','full_grid_maximum_cell_label','full_grid_range']
    f6[keep].to_csv(figsd/'figure_6_deletion_robustness.csv',index=False)

    main[['role','period','a_CMUR_TERR','b_DMC_TERR','c_RMC_TERR','d_CMUR_CFOOT','e_DMC_CFOOT','f_RMC_CFOOT']].to_csv(figsd/'figure_s1_temporal_concordance_matrices.csv',index=False)
    cc.to_csv(figsd/'figure_s2_geography_representation_shifts.csv',index=False)
    country[['geo','CMUR','DMC','RMC','TERR','CFOOT']].to_csv(figsd/'figure_s3_geography_progress_scores.csv',index=False)

    ps=[]; prow=_primary_grid_row(analysis,'pearson'); srow=_primary_grid_row(analysis,'spearman'); colkeys={'CMUR_TERR':'a_CMUR_TERR','DMC_TERR':'b_DMC_TERR','RMC_TERR':'c_RMC_TERR','CMUR_CFOOT':'d_CMUR_CFOOT','DMC_CFOOT':'e_DMC_CFOOT','RMC_CFOOT':'f_RMC_CFOOT'}
    for panel,key in colkeys.items():
        mat,clim=panel.split('_')
        for _,c0 in country.iterrows(): ps.append([panel,c0.geo,float(c0[mat]),float(c0[clim]),float(srow[key]),float(prow[key])])
    pd.DataFrame(ps,columns=['panel','geo','material_progress_score','climate_progress_score','spearman_rho','pearson_r']).to_csv(figsd/'figure_s4_progress_score_geometry.csv',index=False)

    # S5 selected-endpoint summary, clearly secondary to the full-grid hierarchy.
    f5=[['full_reference','selected_endpoint_cell_difference',float(r.D_TOTAL_f_minus_a),float(r.D_TOTAL_f_minus_a),0]]
    for _,rr0 in robustness[robustness.estimand=='D_TOTAL_f_minus_a'].iterrows():
        if rr0.diagnostic in ('LOCO','LOYO'): f5.append([rr0.diagnostic,'selected_endpoint_cell_difference',float(rr0['min']),float(rr0['max']),int(rr0.sign_reversals)])
    for label,spec,est,mode in [('published_pc','published_pc','OLS','pp'),('common_log_CMUR','primary_common_pc','OLS','log'),('Theil_Sen','primary_common_pc','THEIL_SEN','pp'),('total_scale','total_scale','OLS','pp')]:
        q=full_rg[(full_rg.spec==spec)&(full_rg.estimator==est)&(full_rg.cmur_mode==mode)]
        if len(q)==1: v=float(q.iloc[0].D_TOTAL_f_minus_a); f5.append([label,'selected_endpoint_cell_difference',v,v,0])
    f5.append(['CMUR_even_year','selected_endpoint_cell_difference',float(ce.D_TOTAL_f_minus_a),float(ce.D_TOTAL_f_minus_a),0])
    pd.DataFrame(f5,columns=['diagnostic','estimand','min','max','sign_reversals']).drop_duplicates().to_csv(figsd/'figure_s5_broader_robustness_summary.csv',index=False)

    # S6 point-slope sign-disagreement intersections.
    sf=cc.groupby(['point_sign_disagreement_CMUR_DMC','point_sign_disagreement_DMC_RMC','point_sign_disagreement_TERR_CFOOT'],dropna=False).agg(n_countries=('geo','size'),countries=('geo',lambda x:', '.join(sorted(x)))).reset_index().sort_values(['n_countries','point_sign_disagreement_CMUR_DMC','point_sign_disagreement_DMC_RMC','point_sign_disagreement_TERR_CFOOT'],ascending=[False,True,True,True])
    sf.to_csv(figsd/'figure_s6_sign_flip_overlap.csv',index=False)

    # S7 unchanged country-by-cell rank discordance.
    cd=[]
    for _,c0 in country.iterrows():
        for mat in ['CMUR','DMC','RMC']:
            for clim in ['TERR','CFOOT']: cd.append([c0.geo,f'{mat}×{clim}',float(c0[f'rank_{mat}']-c0[f'rank_{clim}'])])
    pd.DataFrame(cd,columns=['geo','cell','material_minus_climate_rank']).to_csv(figsd/'figure_s7_country_discordance_matrix.csv',index=False)


def _natural_earth_rows(ne_zip: Path):
    with tempfile.TemporaryDirectory(prefix="natural_earth_") as td:
        with zipfile.ZipFile(ne_zip) as z:
            if z.testzip() is not None: raise RuntimeError("Natural Earth ZIP CRC failure")
            z.extractall(td)
        shp=next(Path(td).glob("*.shp"))
        rdr=shapefile.Reader(str(shp),encoding="latin1")
        fields=[f[0] for f in rdr.fields[1:]]
        for sr in rdr.iterShapeRecords():
            rec=dict(zip(fields,list(sr.record)))
            yield rec, shapely_shape(sr.shape.__geo_interface__)


def build_country_code_map(ne_zip: Path, output_root: Path) -> tuple[pd.DataFrame, dict]:
    records=list(_natural_earth_rows(ne_zip))
    mapping=[]; geoms={}
    clip=box(-12.8,34.0,35.8,72.5)
    transformer=Transformer.from_crs("EPSG:4326","EPSG:3035",always_xy=True)
    for sci in EU27:
        if sci=="FR":
            cand=[(r,g) for r,g in records if str(r.get("ADM0_A3"))=="FRA" and str(r.get("ADMIN"))=="France"]
            method="ADM0_A3=FRA + ADMIN=France (avoids French dependency records)"
            ne_a2="FR"
        elif sci=="EL":
            cand=[(r,g) for r,g in records if str(r.get("ISO_A2"))=="GR" and str(r.get("ADMIN"))=="Greece"]
            method="Eurostat EL explicitly mapped to Natural Earth ISO_A2=GR"
            ne_a2="GR"
        else:
            cand=[(r,g) for r,g in records if str(r.get("ISO_A2"))==sci]
            method="Natural Earth ISO_A2 exact match"
            ne_a2=sci
        if len(cand)!=1:
            raise RuntimeError(f"Natural Earth join for {sci} returned {len(cand)} rows")
        r,g=cand[0]
        clipped=g.intersection(clip)
        if clipped.is_empty: raise RuntimeError(f"Natural Earth geometry empty after Europe crop: {sci}")
        proj=shp_transform(transformer.transform,clipped).simplify(2500,preserve_topology=True)
        geoms[sci]=proj
        mapping.append([sci,ne_a2,r.get("ADM0_A3"),r.get("ADMIN"),method,1,0,"PASS"])
    df=pd.DataFrame(mapping,columns=["scientific_geo","natural_earth_iso_a2","natural_earth_adm0_a3","natural_earth_admin","match_method","matched_geometry_n","duplicate_matches_n","status"])
    if len(df)!=27 or df.scientific_geo.nunique()!=27 or (df.status!="PASS").any(): raise RuntimeError("Natural Earth 27/27 join check failed")
    (output_root/"cartography").mkdir(parents=True,exist_ok=True)
    df.to_csv(output_root/"cartography"/"COUNTRY_CODE_MAP.csv",index=False)
    parity=df.copy()
    parity["expected_eu27_n"]=27; parity["matched_eu27_n"]=27; parity["missing_eu27_n"]=0; parity["duplicate_country_matches_n"]=0; parity["scientific_value_recomputed"]=False
    parity.to_csv(output_root/"MAP_JOIN_PARITY.csv",index=False)
    return df, geoms


def _poly_patches(geom, **kwargs):
    geoms=list(geom.geoms) if geom.geom_type=="MultiPolygon" else [geom]
    return [MplPolygon(np.asarray(p.exterior.coords),closed=True,**kwargs) for p in geoms if not p.is_empty]


def _map_extent(geoms):
    bounds=np.array([g.bounds for g in geoms.values()])
    xmin,ymin,xmax,ymax=bounds[:,0].min(),bounds[:,1].min(),bounds[:,2].max(),bounds[:,3].max()
    dx=xmax-xmin; dy=ymax-ymin
    return xmin-.025*dx,xmax+.025*dx,ymin-.02*dy,ymax+.02*dy


def _draw_map(ax, geoms, values, cmap, norm, extent):
    for c in EU27:
        col=cmap(norm(values[c])) if c in values and pd.notna(values[c]) else "#F4F4F4"
        for patch in _poly_patches(geoms[c],facecolor=col,edgecolor="#777777",linewidth=.32): ax.add_patch(patch)
    ax.set_xlim(extent[0],extent[1]); ax.set_ylim(extent[2],extent[3]); ax.set_aspect("equal"); ax.axis("off")


def _save(fig, output_root: Path, stem: str):
    d=output_root/"figures"/"reference"; d.mkdir(parents=True,exist_ok=True)
    meta={"Creator":"eu27-material-climate-progress public renderer","CreationDate":None,"ModDate":None}
    fig.savefig(d/f"{stem}.pdf",bbox_inches="tight",pad_inches=.03,metadata=meta)
    fig.savefig(d/f"{stem}.png",dpi=300,bbox_inches="tight",pad_inches=.03,metadata={"Software":"eu27-material-climate-progress public renderer"})
    fig.savefig(d/f"{stem}.svg",bbox_inches="tight",pad_inches=.03,metadata={"Date":None,"Creator":"eu27-material-climate-progress public renderer"})
    plt.close(fig)


def _publication_labels(output_root: Path | None = None) -> dict:
    # The publication-label dictionary is a repository input, not an output-root artifact.
    # This keeps --output-root usable with a genuinely empty destination.
    d=pd.read_csv(ROOT/'docs'/'PUBLICATION_LABELS.csv',keep_default_na=False)
    return {r.internal_key:{'public':r.public_label,'short':r.short_public_label or r.public_label} for r in d.itertuples()}


def _pub(labels: dict, key: str, short: bool=True) -> str:
    if key not in labels:
        raise KeyError(f'Missing publication label: {key}')
    return labels[key]['short' if short else 'public']


def _adaptive_text(rgba) -> str:
    r,g,b=rgba[:3]
    lum=.2126*r+.7152*g+.0722*b
    return 'black' if lum>.58 else 'white'


def _rr_candidates():
    out=[]
    for r in [6,8,10,12,14,17,20,24,28,34]:
        for a in np.linspace(0,2*np.pi,16,endpoint=False): out.append((r*np.cos(a),r*np.sin(a)))
    return out


def _place_rank_labels(fig,ax,dd,font_pt=4.0):
    fig.canvas.draw(); ren=fig.canvas.get_renderer(); axbox=ax.get_window_extent(ren)
    marker=[]; placed=[]; rows=[]
    for _,r in dd.iterrows():
        px,py=ax.transData.transform((r.material_rank,r.climate_rank)); marker.append((r.geo,Bbox.from_extents(px-3.2,py-3.2,px+3.2,py+3.2)))
    lab=dd.assign(gap=(dd.material_rank-dd.climate_rank).abs()).sort_values(['gap','geo'],ascending=[False,True]); dims={}; tmp=[]
    for _,r in lab.iterrows(): tmp.append((r.geo,ax.text(r.material_rank,r.climate_rank,r.geo,fontsize=font_pt,ha='center',va='center',alpha=0)))
    fig.canvas.draw(); ren=fig.canvas.get_renderer()
    for c,t in tmp: b=t.get_window_extent(ren); dims[c]=(b.width*1.05,b.height*1.1); t.remove()
    overrides={
        ('CMUR_TERR','PL'):[(-8,-8)],('CMUR_TERR','LU'):[(-8,-7)],('DMC_TERR','LT'):[(8,7)],('DMC_TERR','HU'):[(-8,-8)],('DMC_TERR','DK'):[(8,-7)],('DMC_TERR','EL'):[(-8,7)],('DMC_TERR','NL'):[(-8,-7)],('RMC_TERR','HU'):[(-8,-8)],('RMC_TERR','BE'):[(-8,7)],('CMUR_CFOOT','LU'):[(-8,-7)],('DMC_CFOOT','HR'):[(8,7)],('DMC_CFOOT','HU'):[(-8,-8)],('DMC_CFOOT','NL'):[(-8,-7)],('DMC_CFOOT','PT'):[(-8,7)],('DMC_CFOOT','SE'):[(8,-7)],('RMC_CFOOT','ES'):[(-8,7)],('RMC_CFOOT','CY'):[(8,-7)]}
    panel=str(dd.panel.iloc[0]); texts=[]
    for _,r in lab.iterrows():
        anchor=np.array(ax.transData.transform((r.material_rank,r.climate_rank))); w,h=dims[r.geo]; chosen=None
        for dx,dy in overrides.get((panel,r.geo),[])+_rr_candidates():
            cp=anchor+np.array([dx,dy]); b=Bbox.from_bounds(cp[0]-w/2,cp[1]-h/2,w,h)
            if b.x0<axbox.x0+1 or b.x1>axbox.x1-1 or b.y0<axbox.y0+1 or b.y1>axbox.y1-1: continue
            if any(Bbox.overlaps(b,pb) for _,pb in placed): continue
            if any(Bbox.overlaps(b,mb) for _,mb in marker): continue
            chosen=(b,ax.transData.inverted().transform(cp),dx,dy); break
        if chosen is None: raise RuntimeError(f'Label placement failed: {panel}/{r.geo}')
        b,data,dx,dy=chosen
        t=ax.text(data[0],data[1],r.geo,fontsize=font_pt,ha='center',va='center',color=DARK,zorder=6,clip_on=True); t._geo=r.geo; texts.append(t); placed.append((r.geo,b))
        rows.append(dict(figure='figure_2_rank_concordance_geometry',panel=panel,country=r.geo,anchor_x=float(r.material_rank),anchor_y=float(r.climate_rank),label_center_x=float(data[0]),label_center_y=float(data[1]),distance_px=float(math.hypot(dx,dy)),label_vs_other_point=0,label_vs_other_label=0,outside_axes=0,status='PASS'))
    fig.canvas.draw(); ren=fig.canvas.get_renderer(); boxes={t._geo:t.get_window_extent(ren).expanded(1.01,1.04) for t in texts}
    for row in rows:
        b=boxes[row['country']]
        row['label_vs_other_point']=int(any(Bbox.overlaps(b,mb) for c,mb in marker if c!=row['country']))
        row['label_vs_other_label']=int(any(Bbox.overlaps(b,ob) for c,ob in boxes.items() if c!=row['country']))
        row['outside_axes']=int(not(axbox.contains(b.x0,b.y0) and axbox.contains(b.x1,b.y1)))
        if row['label_vs_other_point'] or row['label_vs_other_label'] or row['outside_axes']: row['status']='NEEDS_FIX'
    return rows


def _s2_label_offsets():
    # Directional micro-overrides in typographic points. Text alignment is chosen
    # from the sign of the offset, so a small offset keeps labels close to their
    # own point-sign marker anchor rather than shifting the text center across borders.
    return {
        'CMUR → DMC':{
            'SE':(0,3),'FI':(0,-4),'EE':(6,1),'LV':(5,0),'LT':(6,-3),'DK':(-4,0),'ES':(0,4),'FR':(-5,0),
            # Manual Natural Earth label offsets preserve country ownership and legibility.
            'HU':(1,-1),'SI':(-3,0),'HR':(1.5,2),'BG':(6,0),'MT':(5,3)},
        'DMC → RMC':{
            'SE':(0,3),'DE':(0,4),
            'CZ':(-3,0),'SK':(3,0),'SI':(-4,-1),'LU':(-12,4)
        },
        'Territorial → consumption-based GHG':{
            'IE':(4,0),'LV':(5,0),'HU':(-1,-1),'RO':(3,0),'BG':(6,0)
        }
    }


def _render_s2_labels(fig,ax,geoms,d,flip,title):
    offsets=_s2_label_offsets();
    # Short leader lines improve semantic ownership in small-state and dense-country clusters.
    leaders={('CMUR → DMC','MT'),('CMUR → DMC','SI'),('CMUR → DMC','HR'),
             ('DMC → RMC','LU'),('DMC → RMC','SI')}
    q=d[d[flip].astype(bool)].copy().sort_values('geo'); marker_boxes={}; texts=[]; rows=[]
    for _,r0 in q.iterrows():
        p=geoms[r0.geo].representative_point(); px,py=ax.transData.transform((p.x,p.y)); marker_boxes[r0.geo]=Bbox.from_extents(px-3.0,py-3.0,px+3.0,py+3.0)
        ax.scatter([p.x],[p.y],marker='D',s=13,facecolors='none',edgecolors='black',linewidths=.62,zorder=6)
    for _,r0 in q.iterrows():
        c=r0.geo; p=geoms[c].representative_point(); dx,dy=offsets.get(title,{}).get(c,(4,3))
        ha='left' if dx>0 else ('right' if dx<0 else 'center'); va='bottom' if dy>0 else ('top' if dy<0 else 'center')
        leader=(title,c) in leaders
        t=ax.annotate(c,xy=(p.x,p.y),xytext=(dx,dy),textcoords='offset points',fontsize=4.0,ha=ha,va=va,color='#222222',zorder=8,clip_on=True)
        if leader:
            anchor_disp=np.array(ax.transData.transform((p.x,p.y))); text_anchor_disp=anchor_disp+np.array([dx,dy])*(fig.dpi/72.0); data_anchor=ax.transData.inverted().transform(text_anchor_disp)
            ax.plot([p.x,float(data_anchor[0])],[p.y,float(data_anchor[1])],color='#8F8F8F',lw=.22,zorder=5,solid_capstyle='round')
        t._geo=c; texts.append(t)
        rows.append(dict(figure='figure_s2_geography_representation_shifts',panel=title,country=c,marker_anchor_x=float(p.x),marker_anchor_y=float(p.y),text_bbox_x0_px=np.nan,text_bbox_y0_px=np.nan,text_bbox_x1_px=np.nan,text_bbox_y1_px=np.nan,label_center_x=np.nan,label_center_y=np.nan,distance_px=np.nan,leader='YES' if leader else 'NO',text_vs_own_marker=0,text_vs_other_marker=0,text_vs_other_text=0,manual_override='YES' if c in offsets.get(title,{}) else 'NO',human_semantic_ownership='PENDING',status='PASS'))
    fig.canvas.draw(); ren=fig.canvas.get_renderer(); axbox=ax.get_window_extent(ren); boxes={t._geo:t.get_window_extent(ren).expanded(1.01,1.04) for t in texts}
    for row in rows:
        c=row['country']; b=boxes[c]; row['text_bbox_x0_px']=float(b.x0); row['text_bbox_y0_px']=float(b.y0); row['text_bbox_x1_px']=float(b.x1); row['text_bbox_y1_px']=float(b.y1); center=np.array([(b.x0+b.x1)/2,(b.y0+b.y1)/2]); p=geoms[c].representative_point(); anchor=np.array(ax.transData.transform((p.x,p.y))); data=ax.transData.inverted().transform(center)
        row['label_center_x']=float(data[0]); row['label_center_y']=float(data[1]); row['distance_px']=float(np.linalg.norm(center-anchor))
        row['text_vs_own_marker']=int(Bbox.overlaps(b,marker_boxes[c]))
        row['text_vs_other_marker']=int(any(Bbox.overlaps(b,mb) for oc,mb in marker_boxes.items() if oc!=c))
        row['text_vs_other_text']=int(any(Bbox.overlaps(b,ob) for oc,ob in boxes.items() if oc!=c))
        outside=int(not(axbox.contains(b.x0,b.y0) and axbox.contains(b.x1,b.y1)))
        if row['text_vs_own_marker'] or row['text_vs_other_marker'] or row['text_vs_other_text'] or outside: row['status']='NEEDS_FIX'
    return rows


def render_all(output_root: Path, geoms: dict):
    sd=output_root/'figures'/'source_data'; extent=_map_extent(geoms); labels=_publication_labels(output_root); qa={'figure2_labels':[],'figure3_glyphs':[],'figure_s2_labels':[]}
    terr=_pub(labels,'TERR'); cfoot=_pub(labels,'CFOOT')

    # Figure 1 — selected compact matrix grammar: climate rows × material columns, sequential viridis.
    d=pd.read_csv(sd/'figure_1_primary_concordance_grid.csv'); mats=['CMUR','DMC','RMC']; clims=[terr,cfoot]
    arr=np.array([[float(d[(d.material_representation==m)&(d.climate_perspective==c)].spearman_rho.iloc[0]) for m in mats] for c in clims])
    fig,ax=plt.subplots(figsize=(W85,2.55)); cmap=plt.get_cmap('viridis'); norm=Normalize(0,.72)
    mesh=ax.pcolormesh(np.arange(4)-.5,np.arange(3)-.5,arr,cmap=cmap,norm=norm,shading='flat',edgecolors='white',linewidth=.65); ax.set_ylim(1.5,-.5)
    ax.set_xticks(range(3),mats); ax.set_yticks(range(2),clims)
    for i in range(2):
        for j in range(3):
            col=cmap(norm(arr[i,j])); ax.text(j,i,f'{arr[i,j]:.3f}',ha='center',va='center',fontsize=8.0,fontweight='bold',color=_adaptive_text(col))
    [sp.set_visible(False) for sp in ax.spines.values()]; ax.tick_params(length=0)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,fraction=.055,pad=.04); cb.solids.set_rasterized(False); cb.set_label('Spearman ρ'); cb.set_ticks([0,.2,.4,.6,.7])
    fig.subplots_adjust(left=.28,right=.90,bottom=.16,top=.96); _save(fig,output_root,'figure_1_primary_concordance_grid')

    # Figure 2 — all-country rank-geometry design retained.
    d=pd.read_csv(sd/'figure_2_rank_concordance_geometry.csv'); fig,axes=plt.subplots(2,3,figsize=(W178,5.1)); order=['CMUR_TERR','DMC_TERR','RMC_TERR','CMUR_CFOOT','DMC_CFOOT','RMC_CFOOT']; letters='ABCDEF'
    for idx,(ax,panel) in enumerate(zip(axes.ravel(),order)):
        q=d[d.panel==panel].copy(); rho=float(q.spearman_rho.iloc[0]); ax.plot([1,27],[1,27],color='#BDBDBD',lw=.6,zorder=1); ax.scatter(q.material_rank,q.climate_rank,s=7,color='#4F5B66',zorder=2)
        ax.set_xlim(.1,27.9); ax.set_ylim(.1,27.9); ax.set_xticks([1,7,14,21,27]); ax.set_yticks([1,7,14,21,27]); ax.grid(color='#F1F1F1',lw=.35); ax.set_title(f'{q.panel_label.iloc[0]}\nρ = {rho:.3f}',fontsize=6.2,pad=4); ax.text(-.08,1.04,letters[idx],transform=ax.transAxes,fontsize=7,fontweight='bold',ha='left',va='bottom',clip_on=False)
        qa['figure2_labels'] += _place_rank_labels(fig,ax,q,4.0)
    fig.supxlabel('Material progress rank (1 = most favorable)',fontsize=7); fig.supylabel('Climate progress rank (1 = most favorable)',fontsize=7); fig.subplots_adjust(left=.065,right=.995,bottom=.08,top=.96,wspace=.14,hspace=.34); _save(fig,output_root,'figure_2_rank_concordance_geometry')

    # Figure 3 — selected country-consequence cell values with non-overlapping corner sign-disagreement glyphs.
    d=pd.read_csv(sd/'figure_3_country_representation_consequences.csv').sort_values('geo'); cols=['rank_shift_CMUR_to_DMC','rank_shift_DMC_to_RMC','rank_shift_TERR_to_CFOOT']; xlabels=['CMUR → DMC','DMC → RMC','Territorial →\nconsumption-based GHG']; flips=['point_sign_disagreement_CMUR_DMC','point_sign_disagreement_DMC_RMC','point_sign_disagreement_TERR_CFOOT']
    arr=d[cols].to_numpy(float); mx=max(abs(arr.min()),abs(arr.max())); fig,ax=plt.subplots(figsize=(W85,5.65)); norm=TwoSlopeNorm(vmin=-mx,vcenter=0,vmax=mx)
    mesh=ax.pcolormesh(np.arange(4)-.5,np.arange(len(d)+1)-.5,arr,cmap=DIVERGE,norm=norm,shading='flat',edgecolors='white',linewidth=.42); ax.set_ylim(len(d)-.5,-.5); ax.set_yticks(np.arange(len(d)),d.geo); ax.set_xticks(range(3),xlabels); ax.tick_params(axis='x',length=0,labelsize=5.25); ax.tick_params(axis='y',length=0)
    text_objs={}; glyph_boxes=[]
    for i,(_,r0) in enumerate(d.iterrows()):
        for j in range(3):
            rgba=DIVERGE(norm(arr[i,j])); t=ax.text(j,i,f'{arr[i,j]:+.0f}',ha='center',va='center',fontsize=5.15,color=_adaptive_text(rgba),zorder=5); text_objs[(i,j)]=t
            if bool(r0[flips[j]]):
                tri=MplPolygon([[j+.19,i-.46],[j+.46,i-.46],[j+.46,i-.19]],closed=True,facecolor='black',edgecolor='none',zorder=6); ax.add_patch(tri); glyph_boxes.append((i,j,tri))
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=DIVERGE),ax=ax,fraction=.032,pad=.025); cb.solids.set_rasterized(False); cb.set_label('Signed rank displacement')
    da=DrawingArea(10,10,0,0)
    da.add_artist(MplPolygon([[4,10],[10,10],[10,4]],closed=True,facecolor='black',edgecolor='none'))
    ab=AnnotationBbox(da,(.030,-.096),xycoords=ax.transAxes,frameon=False,box_alignment=(0,.5),pad=0,annotation_clip=False,zorder=7)
    ax.add_artist(ab)
    ax.text(.135,-.096,'corner mark = point-slope sign disagreement',transform=ax.transAxes,fontsize=5.3,ha='left',va='center',clip_on=False)
    fig.subplots_adjust(left=.14,right=.88,bottom=.145,top=.985); fig.canvas.draw(); ren=fig.canvas.get_renderer()
    for i,j,tri in glyph_boxes:
        tb=text_objs[(i,j)].get_window_extent(ren).expanded(1.02,1.05); verts=ax.transData.transform(tri.get_xy()); gb=Bbox.from_extents(verts[:,0].min(),verts[:,1].min(),verts[:,0].max(),verts[:,1].max()); ov=int(Bbox.overlaps(tb,gb)); qa['figure3_glyphs'].append(dict(country=d.iloc[i].geo,substitution=xlabels[j],glyph_vs_number_overlap=ov,status='PASS' if ov==0 else 'NEEDS_FIX'))
    _save(fig,output_root,'figure_3_country_representation_consequences')

    # Figure 4 — selected 3+2 layout on frozen Natural Earth geometry.
    d=pd.read_csv(sd/'figure_4_geography_progress_ranks.csv'); rankcols=[('rank_CMUR','CMUR'),('rank_DMC','DMC'),('rank_RMC','RMC'),('rank_TERR',terr),('rank_CFOOT',cfoot)]; rn=Normalize(1,27); rcmap=plt.get_cmap('viridis_r')
    fig=plt.figure(figsize=(W178,5.05)); gs=fig.add_gridspec(4,6,height_ratios=[.14,1,.18,1],left=.002,right=.998,bottom=.105,top=.982,wspace=.002,hspace=.010)
    h1=fig.add_subplot(gs[0,:]); h1.axis('off'); h1.text(.5,.58,'Material representations',ha='center',va='center',fontsize=6.6,fontweight='semibold')
    axs=[fig.add_subplot(gs[1,0:2]),fig.add_subplot(gs[1,2:4]),fig.add_subplot(gs[1,4:6]),fig.add_subplot(gs[3,1:3]),fig.add_subplot(gs[3,3:5])]
    h2=fig.add_subplot(gs[2,:]); h2.axis('off'); h2.text(.5,.72,'Climate perspectives',ha='center',va='center',fontsize=6.6,fontweight='semibold')
    for ax,(col,title) in zip(axs,rankcols): _draw_map(ax,geoms,dict(zip(d.geo,d[col])),rcmap,rn,extent); ax.set_title(title if len(title)<24 else title.replace(' GHG footprint',' GHG\nfootprint'),fontsize=6.7,pad=.55)
    cax=fig.add_axes([.30,.032,.40,.025]); cb=fig.colorbar(plt.cm.ScalarMappable(norm=rn,cmap=rcmap),cax=cax,orientation='horizontal'); cb.solids.set_rasterized(False); cb.set_ticks([1,7,14,21,27]); cb.set_label('Progress rank (1 = most favorable)')
    fig.canvas.draw()
    for a,b in [(axs[0],axs[1]),(axs[1],axs[2]),(axs[3],axs[4])]:
        x=(a.get_position().x1+b.get_position().x0)/2; y0=max(a.get_position().y0,b.get_position().y0); y1=min(a.get_position().y1,b.get_position().y1); fig.add_artist(Line2D([x,x],[y0,y1],transform=fig.transFigure,color='#C8C8C8',lw=.55,zorder=0))
    _save(fig,output_root,'figure_4_geography_progress_ranks')

    # Figure 5 — temporal line-profile design with publication labels and group hierarchy.
    d=pd.read_csv(sd/'figure_5_temporal_representation_sensitivity.csv').set_index('period').loc[PERIOD_ORDER].reset_index(); x=np.arange(6); cols=['a_CMUR_TERR','b_DMC_TERR','c_RMC_TERR','d_CMUR_CFOOT','e_DMC_CFOOT','f_RMC_CFOOT']; xlab=['CMUR','DMC','RMC','CMUR','DMC','RMC']
    fig,ax=plt.subplots(figsize=(W178,3.15)); styles=[(OKABE_BLUE,'D','-',1.65),(OKABE_ORANGE,'o','--',1.25),('#009E73','^','-.',1.25)]
    for (_,r0),(color,marker,ls,lw) in zip(d.iterrows(),styles): ax.plot(x,[r0[c] for c in cols],marker=marker,ls=ls,lw=lw,color=color,label=PERIOD_LABEL[r0.period],markersize=4)
    ax.axvline(2.5,color='#BBBBBB',lw=.65); ax.axhline(0,color='#8D8D8D',lw=.6); ax.set_xticks(x,xlab); ax.set_ylabel('Spearman ρ'); ax.set_ylim(-.28,.78); ax.grid(axis='y',color=GRID,lw=.4); ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.13)); [ax.spines[s].set_visible(False) for s in ['top','right']]
    ax.text(1,-.18,terr,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=6.2,fontweight='semibold'); ax.text(4,-.18,cfoot,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=6.2,fontweight='semibold')
    fig.subplots_adjust(left=.08,right=.99,bottom=.27,top=.84); _save(fig,output_root,'figure_5_temporal_representation_sensitivity')

    # Figure 6 — deletion robustness of the full six-cell grid range only.
    d=pd.read_csv(sd/'figure_6_deletion_robustness.csv')
    full=float(d.loc[d.source.eq('FULL'),'full_grid_range'].iloc[0])
    dl=d[d.source.eq('LOCO')]['full_grid_range'].astype(float).to_numpy()
    dy=d[d.source.eq('LOYO')]['full_grid_range'].astype(float).to_numpy()
    fig,ax=plt.subplots(figsize=(W178,2.20))
    jl=np.linspace(-.115,.115,len(dl)) if len(dl)>1 else np.array([0.0])
    jy=np.linspace(-.095,.095,len(dy)) if len(dy)>1 else np.array([0.0])
    ax.scatter(dl,0.07+jl,s=14,color=OKABE_BLUE,alpha=.78,label='Leave one country out',zorder=2)
    ax.scatter(dy,-.07+jy,s=17,facecolors='white',edgecolors=OKABE_ORANGE,linewidths=.8,label='Leave one year out',zorder=3)
    ax.scatter([full],[0],marker='D',s=28,color=DARK,label='Full 2010–2023',zorder=5)
    ax.axvline(full,color='#B8B8B8',lw=.65,ls='--',zorder=1)
    ax.set_yticks([0],['Full-grid range'])
    ax.set_ylim(-.34,.34)
    ax.set_xlabel('Full-grid range (max ρ − min ρ)')
    ax.grid(axis='x',color=GRID,lw=.4)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.25),ncol=3,frameon=False,handletextpad=.4,columnspacing=1.1)
    [ax.spines[k].set_visible(False) for k in ['top','right','left']]
    ax.tick_params(axis='y',length=0)
    fig.subplots_adjust(left=.19,right=.985,bottom=.33,top=.96)
    _save(fig,output_root,'figure_6_deletion_robustness')

    # Figure S1 — selected temporal matrices; diverging scale is scientifically warranted.
    d=pd.read_csv(sd/'figure_s1_temporal_concordance_matrices.csv').set_index('period').loc[PERIOD_ORDER]; vals=[]
    for _,r in d.iterrows(): vals += [r.a_CMUR_TERR,r.b_DMC_TERR,r.c_RMC_TERR,r.d_CMUR_CFOOT,r.e_DMC_CFOOT,r.f_RMC_CFOOT]
    norm=TwoSlopeNorm(vmin=min(vals),vcenter=0,vmax=max(vals)); fig=plt.figure(figsize=(W178,2.72)); gs=fig.add_gridspec(1,4,width_ratios=[1,1,1,.055],wspace=.16); titles=['2010–2023\nPrimary','2010–2016\nSecondary','2017–2023\nSecondary']
    for k,(_,r) in enumerate(d.iterrows()):
        ax=fig.add_subplot(gs[0,k]); arr=np.array([[r.a_CMUR_TERR,r.b_DMC_TERR,r.c_RMC_TERR],[r.d_CMUR_CFOOT,r.e_DMC_CFOOT,r.f_RMC_CFOOT]]); ax.set_xlim(-.5,2.5); ax.set_ylim(1.5,-.5)
        for i in range(2):
            for j in range(3):
                col=DIVERGE(norm(arr[i,j])); ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,facecolor=col,edgecolor='white',lw=.7)); ax.text(j,i,f'{arr[i,j]:.3f}',ha='center',va='center',fontsize=6.2,color=_adaptive_text(col),fontweight='semibold')
        ax.set_xticks(range(3),['CMUR','DMC','RMC']); ax.set_title(titles[k],fontsize=6.5,pad=5); ax.tick_params(length=0)
        if k==0: ax.set_yticks([0,1],[terr,cfoot],fontsize=5.8)
        else: ax.set_yticks([])
        [sp.set_visible(False) for sp in ax.spines.values()]
    cax=fig.add_subplot(gs[0,3]); cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=DIVERGE),cax=cax); cb.solids.set_rasterized(False); cb.set_label('Spearman ρ'); fig.subplots_adjust(left=.15,right=.97,bottom=.16,top=.86); _save(fig,output_root,'figure_s1_temporal_concordance_matrices')

    # Figure S2 — Natural Earth country-label adaptation with explicit semantic label policy.
    d=pd.read_csv(sd/'figure_s2_geography_representation_shifts.csv'); panels=[('rank_shift_CMUR_to_DMC','point_sign_disagreement_CMUR_DMC','CMUR → DMC'),('rank_shift_DMC_to_RMC','point_sign_disagreement_DMC_RMC','DMC → RMC'),('rank_shift_TERR_to_CFOOT','point_sign_disagreement_TERR_CFOOT','Territorial → consumption-based GHG')]; mx=max(abs(d[[p[0] for p in panels]].to_numpy(float).min()),abs(d[[p[0] for p in panels]].to_numpy(float).max())); norm=TwoSlopeNorm(vmin=-mx,vcenter=0,vmax=mx)
    fig,axes=plt.subplots(1,3,figsize=(W178,2.75),gridspec_kw={'wspace':.025})
    fig.subplots_adjust(left=.005,right=.995,bottom=.18,top=.94)
    for ax,(col,flip,title) in zip(axes,panels):
        _draw_map(ax,geoms,dict(zip(d.geo,d[col])),DIVERGE,norm,extent); ax.set_title(title,fontsize=6.5,pad=2); qa['figure_s2_labels'] += _render_s2_labels(fig,ax,geoms,d,flip,title)
    cax=fig.add_axes([.30,.06,.40,.025]); cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=DIVERGE),cax=cax,orientation='horizontal'); cb.solids.set_rasterized(False); cb.set_label('Signed rank displacement'); fig.text(.72,.073,'◇ point-slope sign disagreement',fontsize=5.3,ha='left',va='center')
    fig.canvas.draw()
    for a,b in [(axes[0],axes[1]),(axes[1],axes[2])]:
        x=(a.get_position().x1+b.get_position().x0)/2; fig.add_artist(Line2D([x,x],[.20,.91],transform=fig.transFigure,color='#C8C8C8',lw=.5,zorder=0))
    _save(fig,output_root,'figure_s2_geography_representation_shifts')

    # Figure S3 — selected layout discipline on Natural Earth, separate CMUR and common log-rate scales.
    d=pd.read_csv(sd/'figure_s3_geography_progress_scores.csv'); cols=[('CMUR','CMUR'),('DMC','DMC'),('RMC','RMC'),('TERR',terr),('CFOOT',cfoot)]; others=d[['DMC','RMC','TERR','CFOOT']].to_numpy(float); lim=max(abs(others.min()),abs(others.max())); cmur_lim=max(abs(d.CMUR.min()),abs(d.CMUR.max())); lnorm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim); cnorm=TwoSlopeNorm(vmin=min(d.CMUR.min(),0),vcenter=0,vmax=max(d.CMUR.max(),0))
    fig=plt.figure(figsize=(W178,5.10)); gs=fig.add_gridspec(4,6,height_ratios=[.14,1,.18,1],left=.002,right=.998,bottom=.117,top=.982,wspace=.002,hspace=.010); h1=fig.add_subplot(gs[0,:]); h1.axis('off'); h1.text(.5,.58,'Material representations',ha='center',va='center',fontsize=6.6,fontweight='semibold'); axs=[fig.add_subplot(gs[1,0:2]),fig.add_subplot(gs[1,2:4]),fig.add_subplot(gs[1,4:6]),fig.add_subplot(gs[3,1:3]),fig.add_subplot(gs[3,3:5])]; h2=fig.add_subplot(gs[2,:]); h2.axis('off'); h2.text(.5,.72,'Climate perspectives',ha='center',va='center',fontsize=6.6,fontweight='semibold')
    for i,(ax,(col,title)) in enumerate(zip(axs,cols)):
        normi=cnorm if i==0 else lnorm; _draw_map(ax,geoms,dict(zip(d.geo,d[col])),DIVERGE,normi,extent); ax.set_title(title if len(title)<24 else title.replace(' GHG footprint',' GHG\nfootprint'),fontsize=6.7,pad=.55)
    c1=fig.add_axes([.09,.037,.31,.024]); cb1=fig.colorbar(plt.cm.ScalarMappable(norm=cnorm,cmap=DIVERGE),cax=c1,orientation='horizontal'); cb1.solids.set_rasterized(False); cb1.set_label('CMUR pp/year')
    c2=fig.add_axes([.54,.037,.37,.024]); cb2=fig.colorbar(plt.cm.ScalarMappable(norm=lnorm,cmap=DIVERGE),cax=c2,orientation='horizontal'); cb2.solids.set_rasterized(False); cb2.set_label('Favorable log-rate/year (DMC/RMC/GHG)')
    fig.canvas.draw()
    for a,b in [(axs[0],axs[1]),(axs[1],axs[2]),(axs[3],axs[4])]:
        x=(a.get_position().x1+b.get_position().x0)/2; y0=max(a.get_position().y0,b.get_position().y0); y1=min(a.get_position().y1,b.get_position().y1); fig.add_artist(Line2D([x,x],[y0,y1],transform=fig.transFigure,color='#C8C8C8',lw=.55,zorder=0))
    _save(fig,output_root,'figure_s3_geography_progress_scores')

    # Figure S4 — progress-score geometry, reader-facing panel labels, no ISO labels.
    d=pd.read_csv(sd/'figure_s4_progress_score_geometry.csv'); order=['CMUR_TERR','DMC_TERR','RMC_TERR','CMUR_CFOOT','DMC_CFOOT','RMC_CFOOT']; fig,axes=plt.subplots(2,3,figsize=(W178,4.9)); letters='ABCDEF'; panel_labels={'CMUR_TERR':f'CMUR × {terr}','DMC_TERR':f'DMC × {terr}','RMC_TERR':f'RMC × {terr}','CMUR_CFOOT':f'CMUR × {cfoot}','DMC_CFOOT':f'DMC × {cfoot}','RMC_CFOOT':f'RMC × {cfoot}'}
    for idx,(ax,panel) in enumerate(zip(axes.ravel(),order)):
        q=d[d.panel==panel]; ax.scatter(q.material_progress_score,q.climate_progress_score,s=8,color=OKABE_BLUE,alpha=.78); ax.axhline(0,color='#BBBBBB',lw=.5); ax.axvline(0,color='#BBBBBB',lw=.5); ax.grid(color='#F0F0F0',lw=.3); ax.set_title(f'{panel_labels[panel]}\nρ = {q.spearman_rho.iloc[0]:.3f} | r = {q.pearson_r.iloc[0]:.3f}',fontsize=5.8,pad=3); ax.text(-.08,1.06,letters[idx],transform=ax.transAxes,fontsize=7,fontweight='bold',ha='left',va='bottom',clip_on=False); [ax.spines[s].set_visible(False) for s in ['top','right']]
    fig.supxlabel('Material progress score'); fig.supylabel('Climate progress score'); fig.subplots_adjust(left=.08,right=.99,bottom=.09,top=.96,wspace=.25,hspace=.32); _save(fig,output_root,'figure_s4_progress_score_geometry')

    # Figure S5 — robust structure retained, labels controlled by publication dictionary.
    d=pd.read_csv(sd/'figure_s5_broader_robustness_summary.csv').drop_duplicates('diagnostic',keep='last'); order=['full_reference','LOCO','LOYO','CMUR_even_year','published_pc','common_log_CMUR','Theil_Sen','total_scale']; d=d.set_index('diagnostic').reindex(order).dropna(how='all').reset_index(); label_map={'full_reference':'Full 2010–2023','LOCO':_pub(labels,'LOCO',False),'LOYO':_pub(labels,'LOYO',False),'CMUR_even_year':_pub(labels,'CMUR_even_year',False),'published_pc':_pub(labels,'published_pc',False),'common_log_CMUR':_pub(labels,'common_log_CMUR',False),'Theil_Sen':_pub(labels,'Theil_Sen',False),'total_scale':_pub(labels,'total_scale',False)}
    fig,ax=plt.subplots(figsize=(W85,3.65)); y=np.arange(len(d)); reference=float(d[d.diagnostic=='full_reference']['min'].iloc[0])
    for i,r0 in d.iterrows():
        if abs(r0['max']-r0['min'])<1e-14: ax.scatter([r0['min']],[i],s=18,facecolors='white',edgecolors=OKABE_BLUE,zorder=3)
        else: ax.plot([r0['min'],r0['max']],[i,i],color=OKABE_BLUE,lw=1.35,marker='o',markersize=2.9)
    ax.axvline(reference,color=OKABE_ORANGE,lw=.9,ls='--'); ax.text(reference,1.015,f'Full 2010–2023 reference = {reference:.3f}',transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=5.8,color=OKABE_ORANGE,clip_on=False); ax.set_yticks(y,[label_map[x] for x in d.diagnostic]); ax.invert_yaxis(); ax.set_xlabel('Selected endpoint difference'); ax.grid(axis='x',color=GRID,lw=.4); [ax.spines[s].set_visible(False) for s in ['top','right','left']]; fig.subplots_adjust(left=.42,right=.98,bottom=.14,top=.91); _save(fig,output_root,'figure_s5_broader_robustness_summary')

    # Figure S6 — categorical UpSet structure retained; no artificial numeric x-axis.
    d=pd.read_csv(sd/'figure_s6_sign_flip_overlap.csv').reset_index(drop=True); x=np.arange(len(d)); fig=plt.figure(figsize=(W85,3.3)); gs=fig.add_gridspec(2,1,height_ratios=[1.25,1],hspace=.05); axb=fig.add_subplot(gs[0]); axm=fig.add_subplot(gs[1],sharex=axb); bars=axb.bar(x,d.n_countries,color=OKABE_BLUE,width=.60)
    for rect,val in zip(bars,d.n_countries): axb.text(rect.get_x()+rect.get_width()/2,rect.get_height()+.08,str(int(val)),ha='center',va='bottom',fontsize=5.8)
    axb.set_ylabel('Countries'); axb.set_xticks([]); axb.set_ylim(0,max(d.n_countries)+1.25); axb.grid(axis='y',color=GRID,lw=.35); [axb.spines[s].set_visible(False) for s in ['top','right','bottom']]
    rows=[('CMUR → DMC','point_sign_disagreement_CMUR_DMC'),('DMC → RMC','point_sign_disagreement_DMC_RMC'),('Territorial → consumption-based GHG','point_sign_disagreement_TERR_CFOOT')]
    for yi,(lab,col) in enumerate(rows):
        xs=[]
        for xi,val in enumerate(d[col].astype(bool)): axm.scatter([xi],[yi],s=21,facecolors='#222222' if val else 'white',edgecolors='#777777',linewidths=.55); xs.append(xi if val else None)
        active=[xi for xi,val in enumerate(d[col].astype(bool)) if val]
    for xi in range(len(d)):
        active=[yi for yi,(_,col) in enumerate(rows) if bool(d.loc[xi,col])]
        if len(active)>1: axm.plot([xi,xi],[min(active),max(active)],color='#222222',lw=.7,zorder=0)
    axm.set_yticks(range(3),[r[0] for r in rows]); axm.set_xticks(x,['' for _ in x]); axm.tick_params(axis='x',length=0); axm.set_xlabel('Disagreement pattern'); axm.set_ylim(2.6,-.6); [axm.spines[s].set_visible(False) for s in ['top','right','left']]; fig.subplots_adjust(left=.32,right=.98,bottom=.16,top=.95); _save(fig,output_root,'figure_s6_sign_flip_overlap')

    # Figure S7 — CD01 grouping, public labels and subtle cell boundaries.
    d=pd.read_csv(sd/'figure_s7_country_discordance_matrix.csv'); countries=sorted(d.geo.unique()); cells=['CMUR×TERR','DMC×TERR','RMC×TERR','CMUR×CFOOT','DMC×CFOOT','RMC×CFOOT']; arr=np.array([[float(d[(d.geo==g)&(d.cell==c)].material_minus_climate_rank.iloc[0]) for c in cells] for g in countries]); mx=max(abs(arr.min()),abs(arr.max())); fig,ax=plt.subplots(figsize=(W178,4.75)); norm=TwoSlopeNorm(vmin=-mx,vcenter=0,vmax=mx); mesh=ax.pcolormesh(np.arange(7)-.5,np.arange(len(countries)+1)-.5,arr,cmap=DIVERGE,norm=norm,shading='flat',edgecolors='#E8E8E8',linewidth=.32); ax.set_ylim(len(countries)-.5,-.5); ax.set_yticks(range(len(countries)),countries); ax.set_xticks(range(6),['CMUR','DMC','RMC','CMUR','DMC','RMC']); ax.axvline(2.5,color='#9D9D9D',lw=.75); ax.text(1,1.02,terr,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=6.4,fontweight='semibold',clip_on=False); ax.text(4,1.02,cfoot,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=6.4,fontweight='semibold',clip_on=False); cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=DIVERGE),ax=ax,fraction=.025,pad=.02); cb.solids.set_rasterized(False); cb.set_label('Material rank − climate rank'); fig.subplots_adjust(left=.06,right=.93,bottom=.08,top=.91); _save(fig,output_root,'figure_s7_country_discordance_matrix')
    return qa

def build_all(analysis: Path, output_root: Path, ne_zip: Path):
    build_publication_sources(analysis, output_root)
    _,geoms=build_country_code_map(ne_zip, output_root)
    return render_all(output_root, geoms)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--analysis",type=Path,default=ROOT/"results"/"reference",help="Validated analysis-output root (tables/, supplementary/, source_data/)")
    ap.add_argument("--output-root",type=Path,default=ROOT,help="Root under which publication paths are written")
    ap.add_argument("--natural-earth",type=Path,default=ROOT/"cartography"/"source"/"ne_10m_admin_0_countries.zip")
    args=ap.parse_args()
    build_all(args.analysis,args.output_root,args.natural_earth)
    print("PUBLICATION_LAYER_BUILD_PASS figures=13 main_tables=4 supplementary_tables=9 map_join=27/27")

if __name__=="__main__":
    main()
