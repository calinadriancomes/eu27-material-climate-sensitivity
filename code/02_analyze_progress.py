#!/usr/bin/env python3
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, theilslopes
from checks import EU27, PERIODS, EVEN_YEARS, RANK_DECIMALS

def ols_slope(x,y):
    return float(np.polyfit(np.asarray(x,float),np.asarray(y,float),1)[0])

def ts_slope(x,y):
    return float(theilslopes(np.asarray(y,float),np.asarray(x,float),method="separate")[0])

def progress_table(panel, geos, years, spec="primary_common_pc", estimator="OLS", cmur_mode="pp"):
    if spec=="primary_common_pc":
        cols={"CMUR":"cmur_pct","DMC":"dmc_pc_common_t","RMC":"rmc_pc_common_t","TERR":"terr_ghg_pc_common_t","CFOOT":"cfoot_pc_common_t"}
    elif spec=="published_pc":
        cols={"CMUR":"cmur_pct","DMC":"dmc_pc_published_t","RMC":"rmc_pc_published_t","TERR":"terr_ghg_pc_published_t","CFOOT":"cfoot_pc_published_t"}
    elif spec=="total_scale":
        cols={"CMUR":"cmur_pct","DMC":"dmc_total_t","RMC":"rmc_total_t","TERR":"terr_ghg_total_t","CFOOT":"cfoot_total_t"}
    else:
        raise ValueError(spec)
    f=ols_slope if estimator=="OLS" else ts_slope
    rows=[]
    for geo in geos:
        g=panel[(panel.geo==geo)&panel.year.isin(years)].sort_values("year")
        if list(pd.to_numeric(g.year).astype(int)) != list(years):
            raise RuntimeError(f"{geo}/{spec}: incomplete common annual support")
        r={"geo":geo,"spec":spec,"estimator":estimator,"cmur_mode":cmur_mode,"years":f"{min(years)}-{max(years)}","n_years":len(years)}
        x=pd.to_numeric(g.year).to_numpy(float)
        for k,c in cols.items():
            v=pd.to_numeric(g[c],errors="coerce").to_numpy(float)
            if np.isnan(v).any():
                raise RuntimeError(f"{geo}/{c}: missing")
            if k=="CMUR":
                if cmur_mode=="pp":
                    y=v
                elif cmur_mode=="log":
                    if np.any(v<=0): raise RuntimeError("Non-positive CMUR")
                    y=np.log(v)
                else:
                    raise ValueError(cmur_mode)
                r[k]=f(x,y)
            else:
                if np.any(v<=0):
                    raise RuntimeError(f"Non-positive {c}")
                r[k]=-f(x,np.log(v))
        rows.append(r)
    return pd.DataFrame(rows)

def rank_canon(x):
    return np.round(np.asarray(x,float), RANK_DECIMALS)

def corr(x,y,method):
    a=np.asarray(x,float); b=np.asarray(y,float)
    if method in ("spearman","kendall"):
        a=rank_canon(a); b=rank_canon(b)
    if method=="spearman": return float(spearmanr(a,b).statistic)
    if method=="kendall": return float(kendalltau(a,b,variant="b").statistic)
    if method=="pearson": return float(np.corrcoef(a,b)[0,1])
    raise ValueError(method)

def grid(prog,method="spearman"):
    a=corr(prog.CMUR,prog.TERR,method); b=corr(prog.DMC,prog.TERR,method); c=corr(prog.RMC,prog.TERR,method)
    d=corr(prog.CMUR,prog.CFOOT,method); e=corr(prog.DMC,prog.CFOOT,method); f=corr(prog.RMC,prog.CFOOT,method)
    D1=b-a; D2=c-b; D3=f-c; DT=f-a
    I1=(e-d)-(b-a); I2=(f-e)-(c-b)
    return {"a_CMUR_TERR":a,"b_DMC_TERR":b,"c_RMC_TERR":c,"d_CMUR_CFOOT":d,"e_DMC_CFOOT":e,"f_RMC_CFOOT":f,
            "D1_b_minus_a":D1,"D2_c_minus_b":D2,"D3_f_minus_c":D3,"D_TOTAL_f_minus_a":DT,
            "I1_construct_x_climate":I1,"I2_scope_x_climate":I2,"identity_D_sum_error":D1+D2+D3-DT}

def country_consequences(prog):
    z=prog[["geo","CMUR","DMC","RMC","TERR","CFOOT"]].copy().set_index("geo")
    for c in ["CMUR","DMC","RMC","TERR","CFOOT"]:
        can=rank_canon(z[c])
        z[f"rank_{c}"]=pd.Series(can,index=z.index).rank(method="average",ascending=False)
        z[f"favourable_{c}"]=z[c]>0
    z["rank_shift_CMUR_to_DMC"]=z.rank_DMC-z.rank_CMUR
    z["rank_shift_DMC_to_RMC"]=z.rank_RMC-z.rank_DMC
    z["rank_shift_TERR_to_CFOOT"]=z.rank_CFOOT-z.rank_TERR
    z["sign_flip_CMUR_DMC"]=z.favourable_CMUR!=z.favourable_DMC
    z["sign_flip_DMC_RMC"]=z.favourable_DMC!=z.favourable_RMC
    z["sign_flip_TERR_CFOOT"]=z.favourable_TERR!=z.favourable_CFOOT
    return z.reset_index().sort_values("geo").reset_index(drop=True)

def all_grid(panel):
    rows=[]
    for per,years in PERIODS.items():
        primary=progress_table(panel,EU27,years)
        for m in ["spearman","kendall","pearson"]:
            r=grid(primary,m); r.update({"period":per,"spec":"primary_common_pc","estimator":"OLS","cmur_mode":"pp","correlation":m}); rows.append(r)
        pub=progress_table(panel,EU27,years,"published_pc")
        for m in ["spearman","kendall","pearson"]:
            r=grid(pub,m); r.update({"period":per,"spec":"published_pc","estimator":"OLS","cmur_mode":"pp","correlation":m}); rows.append(r)
        clog=progress_table(panel,EU27,years,"primary_common_pc","OLS","log")
        r=grid(clog); r.update({"period":per,"spec":"primary_common_pc","estimator":"OLS","cmur_mode":"log","correlation":"spearman"}); rows.append(r)
        ts=progress_table(panel,EU27,years,"primary_common_pc","THEIL_SEN","pp")
        r=grid(ts); r.update({"period":per,"spec":"primary_common_pc","estimator":"THEIL_SEN","cmur_mode":"pp","correlation":"spearman"}); rows.append(r)
        tot=progress_table(panel,EU27,years,"total_scale","OLS","pp")
        r=grid(tot); r.update({"period":per,"spec":"total_scale","estimator":"OLS","cmur_mode":"pp","correlation":"spearman"}); rows.append(r)
    return pd.DataFrame(rows)

def main_hierarchy(g):
    q=g[(g.spec=="primary_common_pc")&(g.estimator=="OLS")&(g.cmur_mode=="pp")&(g.correlation=="spearman")].copy()
    q=q.drop(columns=["identity_D_sum_error","spec","estimator","cmur_mode","correlation"])
    order=["FULL_2010_2023","EARLY_2010_2016","LATE_2017_2023"]
    q["period"]=pd.Categorical(q.period,categories=order,ordered=True)
    q=q.sort_values("period").reset_index(drop=True)
    q.insert(0,"role",["PRIMARY","SECONDARY","SECONDARY"])
    q["period"]=q.period.astype(str)
    cols=["role","period"]+[c for c in q.columns if c not in {"role","period"}]
    return q[cols]

def country_summary(cc):
    rows=[]
    for label,col,flip in [
        ("CMUR_to_DMC","rank_shift_CMUR_to_DMC","sign_flip_CMUR_DMC"),
        ("DMC_to_RMC","rank_shift_DMC_to_RMC","sign_flip_DMC_RMC"),
        ("TERR_to_CFOOT","rank_shift_TERR_to_CFOOT","sign_flip_TERR_CFOOT")]:
        a=cc[col].abs(); mx=float(a.max()); geo=str(cc.loc[a.idxmax(),"geo"])
        rows.append({"substitution":label,"mean_abs_rank_displacement":float(a.mean()),
                     "max_abs_rank_displacement":mx,"max_country":geo,"sign_flips_n":int(cc[flip].sum())})
    return pd.DataFrame(rows)

def cprod_bridge(panel):
    rows=[]
    for per,years in PERIODS.items():
        base=progress_table(panel,EU27,years)
        cp=[]
        for geo in EU27:
            g=panel[(panel.geo==geo)&panel.year.isin(years)].sort_values("year")
            v=pd.to_numeric(g.cprod_pc_common_t).to_numpy(float)
            cp.append({"geo":geo,"C_PROD":-ols_slope(pd.to_numeric(g.year).to_numpy(float),np.log(v))})
        cp=pd.DataFrame(cp).set_index("geo"); b=base.set_index("geo")
        for material in ["CMUR","DMC","RMC"]:
            rt=corr(b[material],b.TERR,"spearman"); rp=corr(b[material],cp.C_PROD,"spearman"); rf=corr(b[material],b.CFOOT,"spearman")
            rows.append({"period":per,"material_representation":material,"rho_material_TERR":rt,"rho_material_PROD":rp,
                         "rho_material_CFOOT":rf,"delta_TERR_to_PROD":rp-rt,"delta_PROD_to_CFOOT":rf-rp,"delta_TERR_to_CFOOT":rf-rt})
    return pd.DataFrame(rows)

def rmc_status(panel):
    rows=[]
    for geo,g in panel.groupby("geo",sort=True):
        s=g.rmc_total_status.fillna("").astype(str)
        iy=list(pd.to_numeric(g.loc[s=="i","year"]).astype(int))
        rows.append({"geo":geo,"years_n":len(g),"flag_i_n":int((s=="i").sum()),"blank_n":int((s=="").sum()),
                     "other_flag_n":int((~s.isin(["","i"])).sum()),"flag_i_years":";".join(map(str,iy)) if iy else np.nan})
    return pd.DataFrame(rows).sort_values(["flag_i_n","geo"],ascending=[False,True]).reset_index(drop=True)

def robustness(panel,g):
    rows=[]; years=PERIODS["FULL_2010_2023"]; full=grid(progress_table(panel,EU27,years))
    loco=[]
    for omit in EU27:
        rec=grid(progress_table(panel,[x for x in EU27 if x!=omit],years)); rec["omit_geo"]=omit; loco.append(rec)
    loyo=[]
    for omit in years:
        rec=grid(progress_table(panel,EU27,[y for y in years if y!=omit])); rec["omit_year"]=omit; loyo.append(rec)
    locodf=pd.DataFrame(loco); loyodf=pd.DataFrame(loyo)
    for est in ["D1_b_minus_a","D2_c_minus_b","D3_f_minus_c","D_TOTAL_f_minus_a"]:
        for label,arr in [("LOCO",locodf),("LOYO",loyodf)]:
            vals=arr[est].to_numpy(float)
            rows.append({"diagnostic":label,"estimand":est,"min":float(vals.min()),"max":float(vals.max()),
                         "sign_reversals":int(np.sum(np.sign(vals)!=np.sign(full[est])))})
    ev=grid(progress_table(panel,EU27,EVEN_YEARS))
    rows.append({"diagnostic":"CMUR_even_year","estimand":"D_TOTAL_f_minus_a","min":ev["D_TOTAL_f_minus_a"],"max":ev["D_TOTAL_f_minus_a"],"sign_reversals":0})
    def pick(spec,est,mode):
        q=g[(g.period=="FULL_2010_2023")&(g.spec==spec)&(g.estimator==est)&(g.cmur_mode==mode)&(g.correlation=="spearman")]
        return float(q.iloc[0].D_TOTAL_f_minus_a)
    for label,spec,est,mode in [("published_pc","published_pc","OLS","pp"),("common_log_CMUR","primary_common_pc","OLS","log"),
                                ("Theil_Sen","primary_common_pc","THEIL_SEN","pp"),("total_scale","total_scale","OLS","pp")]:
        v=pick(spec,est,mode); rows.append({"diagnostic":label,"estimand":"D_TOTAL_f_minus_a","min":v,"max":v,"sign_reversals":0})
    return pd.DataFrame(rows)

def tie_audit(panel):
    rows=[]
    specs=[("primary_common_pc","OLS","pp"),("published_pc","OLS","pp"),("primary_common_pc","THEIL_SEN","pp"),("total_scale","OLS","pp")]
    for per,years in PERIODS.items():
        for spec,est,mode in specs:
            p=progress_table(panel,EU27,years,spec,est,mode)
            raw=p.CMUR.to_numpy(float); can=rank_canon(raw)
            tmp=pd.DataFrame({"geo":p.geo,"raw":raw,"canon":can})
            for cv,x in tmp.groupby("canon"):
                if len(x)>1:
                    rows.append({"period":per,"spec":spec,"estimator":est,"cmur_mode":mode,"variable":"CMUR",
                                 "canonical_value_12dp":cv,"n_tied":len(x),"geos":";".join(sorted(x.geo)),
                                 "raw_min":float(x.raw.min()),"raw_max":float(x.raw.max()),"raw_span":float(x.raw.max()-x.raw.min())})
    return pd.DataFrame(rows)

def detailed_sensitivity_outputs(panel, out_dir):
    supp=out_dir/"supplementary"; supp.mkdir(parents=True,exist_ok=True)
    prows=[]; crows=[]; rank_summ=[]; rank_detail=[]
    for per,years in PERIODS.items():
        specs=[("primary_common_pc","OLS","pp","primary"),("published_pc","OLS","pp","published_pc"),
               ("primary_common_pc","OLS","log","common_log_CMUR"),("primary_common_pc","THEIL_SEN","pp","Theil_Sen"),
               ("total_scale","OLS","pp","total_scale")]
        saved={}
        for spec,est,mode,label in specs:
            pr=progress_table(panel,EU27,years,spec,est,mode); pr.insert(0,"period",per); pr.insert(1,"analysis_label",label); prows.append(pr)
            cc=country_consequences(pr); cc.insert(0,"period",per); cc.insert(1,"analysis_label",label); crows.append(cc); saved[label]=pr
        c=saved["primary"].set_index("geo"); q=saved["published_pc"].set_index("geo")
        det=pd.DataFrame({"period":per,"geo":sorted(c.index)}).set_index("geo")
        for v in ["DMC","RMC","TERR","CFOOT"]:
            rc=pd.Series(rank_canon(c[v]),index=c.index).rank(method="average",ascending=False)
            rp=pd.Series(rank_canon(q[v]),index=q.index).rank(method="average",ascending=False)
            det[f"rank_common_{v}"]=rc; det[f"rank_published_{v}"]=rp; det[f"rank_displacement_{v}"]=rp-rc
            rank_summ.append({"period":per,"representation":v,"spearman_progress_common_vs_published":corr(c[v],q[v],"spearman"),
                              "mean_abs_rank_displacement":float((rp-rc).abs().mean()),"max_abs_rank_displacement":float((rp-rc).abs().max())})
        rank_detail.append(det.reset_index())
    pd.concat(prows,ignore_index=True).to_csv(supp/"progress_scores_all_periods_specs.csv",index=False)
    pd.concat(crows,ignore_index=True).to_csv(supp/"country_consequences_all_periods_specs.csv",index=False)
    pd.DataFrame(rank_summ).to_csv(supp/"published_pc_vs_common_pc_rank_summary.csv",index=False)
    pd.concat(rank_detail,ignore_index=True).to_csv(supp/"published_pc_vs_common_pc_rank_detail.csv",index=False)
    loco=[]
    for per,years in PERIODS.items():
        full=grid(progress_table(panel,EU27,years))
        for omit in EU27:
            rec=grid(progress_table(panel,[g for g in EU27 if g!=omit],years)); row={"period":per,"omit_geo":omit}; row.update(rec)
            for k in ["D1_b_minus_a","D2_c_minus_b","D3_f_minus_c","D_TOTAL_f_minus_a","I1_construct_x_climate","I2_scope_x_climate"]:
                row[f"influence__{k}"]=full[k]-rec[k]
            loco.append(row)
    pd.DataFrame(loco).to_csv(supp/"LOCO_representation_grid.csv",index=False)
    loyo=[]
    for per,years in PERIODS.items():
        for omit in years:
            rec=grid(progress_table(panel,EU27,[y for y in years if y!=omit])); rec.update({"period":per,"omit_year":omit,"n_years":len(years)-1}); loyo.append(rec)
    pd.DataFrame(loyo).to_csv(supp/"LOYO_representation_grid.csv",index=False)
    ev=grid(progress_table(panel,EU27,EVEN_YEARS)); ev.update({"years":";".join(map(str,EVEN_YEARS)),"purpose":"reduce dependence on systematic CMUR odd-year gap-filling"})
    pd.DataFrame([ev]).to_csv(supp/"CMUR_even_year_common_support_grid.csv",index=False)



def write_analysis(panel, out_dir):
    out_dir=Path(out_dir); tables=out_dir/"tables"; supp=out_dir/"supplementary"; source=out_dir/"source_data"
    for d in [tables,supp,source]: d.mkdir(parents=True,exist_ok=True)
    g=all_grid(panel); hierarchy=main_hierarchy(g)
    full_progress=progress_table(panel,EU27,PERIODS["FULL_2010_2023"])
    changes=country_consequences(full_progress); change_summary=country_summary(changes)
    bridge=cprod_bridge(panel); status=rmc_status(panel); robust=robustness(panel,g); ties=tie_audit(panel)
    hierarchy.to_csv(tables/"main_results.csv",index=False)
    g.to_csv(tables/"representation_grid.csv",index=False)
    changes.to_csv(tables/"country_rank_changes.csv",index=False)
    change_summary.to_csv(tables/"country_rank_change_summary.csv",index=False)
    bridge.to_csv(supp/"accounting_bridge.csv",index=False)
    status.to_csv(supp/"rmc_status.csv",index=False)
    robust.to_csv(supp/"robustness_summary.csv",index=False)
    ties.to_csv(supp/"tie_audit.csv",index=False)
    detailed_sensitivity_outputs(panel,out_dir)
    # Rename the detailed files to concise public names.
    rename={
      "progress_scores_all_periods_specs.csv":"progress_scores_all_specs.csv",
      "country_consequences_all_periods_specs.csv":"country_results_all_specs.csv",
      "published_pc_vs_common_pc_rank_summary.csv":"published_vs_common_rank_summary.csv",
      "published_pc_vs_common_pc_rank_detail.csv":"published_vs_common_rank_detail.csv",
      "LOCO_representation_grid.csv":"leave_one_country_out.csv",
      "LOYO_representation_grid.csv":"leave_one_year_out.csv",
      "CMUR_even_year_common_support_grid.csv":"cmur_even_year.csv",
    }
    for old,new in rename.items():
        p=supp/old
        if p.exists(): p.rename(supp/new)
    full=hierarchy[hierarchy.period=="FULL_2010_2023"].iloc[0]
    pd.DataFrame([[full.a_CMUR_TERR,full.d_CMUR_CFOOT],[full.b_DMC_TERR,full.e_DMC_CFOOT],[full.c_RMC_TERR,full.f_RMC_CFOOT]],
                 index=["CMUR","DMC","RMC"],columns=["Territorial GHG","Consumption GHG"]).to_csv(source/"full_representation_grid.csv")
    d=changes[["geo","rank_shift_CMUR_to_DMC","rank_shift_DMC_to_RMC","rank_shift_TERR_to_CFOOT"]].copy()
    for c in d.columns[1:]: d[c]=d[c].abs()
    d.to_csv(source/"country_rank_displacement.csv",index=False)
    hierarchy[["period","D1_b_minus_a","D2_c_minus_b","D3_f_minus_c","D_TOTAL_f_minus_a"]].to_csv(source/"temporal_contrasts.csv",index=False)

def main():
    import argparse, json
    ap=argparse.ArgumentParser(); ap.add_argument("--panel",default=str(ROOT/"work/data/panel_2010_2023.csv")); ap.add_argument("--output",default=str(ROOT/"work/analysis")); args=ap.parse_args()
    panel=pd.read_csv(args.panel,keep_default_na=False)
    if len(panel)!=378 or panel.duplicated(["geo","year"]).any() or set(panel.geo)!=set(EU27): raise RuntimeError("Analysis input must be the complete EU27 panel for 2010–2023")
    write_analysis(panel,Path(args.output))
    print("ANALYSIS_PASS")

if __name__=="__main__": main()
