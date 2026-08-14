#!/usr/bin/env python3
from pathlib import Path
import argparse, gzip, itertools, json, sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from checks import EU27, YEARS, ROUTES, PANEL_SHA256, sha256_file, compare_panel

ROOT=HERE.parent

def category_codes(js, dim):
    cat=js["dimension"][dim]["category"]; idx=cat.get("index",{})
    if isinstance(idx,dict): return [k for k,v in sorted(idx.items(),key=lambda kv:kv[1])]
    if isinstance(idx,list): return list(idx)
    return list(cat.get("label",{}).keys())

def read_jsonstat(path):
    js=json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    dims=list(js.get("id",[])); sizes=[int(x) for x in js.get("size",[])]
    if not dims or len(dims)!=len(sizes): raise RuntimeError(f"{path.name}: invalid JSON-stat dimensions")
    codes=[category_codes(js,d) for d in dims]
    if any(len(codes[i])!=sizes[i] for i in range(len(dims))): raise RuntimeError(f"{path.name}: category size mismatch")
    vals=js.get("value",{}); sts=js.get("status",{}) or {}
    def at(obj,i):
        if isinstance(obj,list): return obj[i] if i<len(obj) else None
        if isinstance(obj,dict): return obj.get(str(i),obj.get(i))
        return None
    rows=[]
    for i,comb in enumerate(itertools.product(*codes)):
        row={d:c for d,c in zip(dims,comb)}; row["value"]=at(vals,i); s=at(sts,i); row["status"]="" if s is None else str(s); rows.append(row)
    return js,pd.DataFrame(rows)

def verify_source_hashes(raw_dir, sources):
    for _,r in sources.iterrows():
        path=raw_dir/r.raw_file
        if not path.is_file(): raise RuntimeError(f"Missing source file: {r.raw_file}")
        if sha256_file(path)!=r.raw_sha256: raise RuntimeError(f"Source checksum mismatch: {r.raw_file}")

def normalize_route(path):
    js,df=read_jsonstat(path)
    if "geo" not in df.columns or "time" not in df.columns: raise RuntimeError(f"{path.name}: geo/time not found")
    q=df[["geo","time","value","status"]].rename(columns={"time":"year"}).copy()
    q["year"]=pd.to_numeric(q.year,errors="coerce")
    q=q[q.geo.isin(EU27)&q.year.isin(YEARS)].copy(); q["year"]=q.year.astype(int)
    q=q.sort_values(["geo","year"]).reset_index(drop=True)
    if q.duplicated(["geo","year"]).any() or len(q)!=378 or q.value.notna().sum()!=378:
        raise RuntimeError(f"{path.name}: expected 378 complete country-years")
    return q

def build_panel(normalized_dir):
    panel=pd.MultiIndex.from_product([EU27,YEARS],names=["geo","year"]).to_frame(index=False)
    for key,(raw_name,value_col,status_col) in ROUTES.items():
        q=pd.read_csv(normalized_dir/f"{key}.csv",keep_default_na=False)
        q=q[["geo","year","value","status"]].rename(columns={"value":value_col,"status":status_col})
        panel=panel.merge(q,on=["geo","year"],how="left",validate="one_to_one")
    panel["dmc_total_t"]=pd.to_numeric(panel.dmc_total_ths_t)*1000.0
    panel["rmc_total_t"]=pd.to_numeric(panel.rmc_total_ths_t)*1000.0
    panel["terr_ghg_total_t"]=pd.to_numeric(panel.terr_ghg_total_raw)*1_000_000.0
    panel["terr_ghg_pc_published_t"]=pd.to_numeric(panel.terr_ghg_pc_published_raw)
    panel["cfoot_total_t"]=pd.to_numeric(panel.cfoot_total_raw)*1000.0
    panel["cfoot_pc_published_t"]=pd.to_numeric(panel.cfoot_pc_published_raw)*0.001
    panel["cprod_total_t"]=pd.to_numeric(panel.cprod_total_raw)*1000.0
    panel["population"]=pd.to_numeric(panel.population)
    if (panel.population<=0).any(): raise RuntimeError("Average population must be positive")
    panel["dmc_pc_common_t"]=panel.dmc_total_t/panel.population
    panel["rmc_pc_common_t"]=panel.rmc_total_t/panel.population
    panel["terr_ghg_pc_common_t"]=panel.terr_ghg_total_t/panel.population
    panel["cfoot_pc_common_t"]=panel.cfoot_total_t/panel.population
    panel["cprod_pc_common_t"]=panel.cprod_total_t/panel.population
    panel=panel.drop(columns=["cprod_total_raw"]).sort_values(["geo","year"]).reset_index(drop=True)
    if len(panel)!=378 or panel.duplicated(["geo","year"]).any(): raise RuntimeError("Panel key check failed")
    return panel

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default=str(ROOT/"work/data")); args=ap.parse_args()
    out=Path(args.output); norm=out/"normalized"; norm.mkdir(parents=True,exist_ok=True)
    raw=ROOT/"data/raw"; sources=pd.read_csv(ROOT/"data/sources.csv",keep_default_na=False)
    verify_source_hashes(raw,sources)
    for key,(raw_name,_,_) in ROUTES.items(): normalize_route(raw/raw_name).to_csv(norm/f"{key}.csv",index=False)
    panel=build_panel(norm); panel_path=out/"panel_2010_2023.csv"; panel.to_csv(panel_path,index=False)
    maxdiff,maxcol=compare_panel(panel_path,ROOT/"data/panel_2010_2023.csv",1e-6)
    validation={"status":"PASS","rows":378,"countries":27,"years":"2010-2023","max_panel_difference":maxdiff,"max_difference_column":maxcol}
    (out/"validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    print(f"DATA_PREPARATION_PASS rows=378 max_panel_difference={maxdiff:.17g}")
if __name__=="__main__": main()
