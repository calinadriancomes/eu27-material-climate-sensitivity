#!/usr/bin/env python3
from pathlib import Path
import importlib.util, shutil, tempfile
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def load_renderer():
    p=ROOT/"code"/"04_render_publication_figures.py"; spec=importlib.util.spec_from_file_location("publication_renderer",p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def compare(a,b,tol=1e-12):
    x=pd.read_csv(a,keep_default_na=False); y=pd.read_csv(b,keep_default_na=False)
    assert list(x.columns)==list(y.columns) and x.shape==y.shape
    for c in x.columns:
        xn=pd.to_numeric(x[c],errors="coerce"); yn=pd.to_numeric(y[c],errors="coerce")
        if xn.notna().sum()==len(x) and yn.notna().sum()==len(y): assert np.max(np.abs(xn.to_numpy(float)-yn.to_numpy(float)))<=tol,c
        else: assert (x[c].astype(str).to_numpy()==y[c].astype(str).to_numpy()).all(),c

def main():
    m=load_renderer(); td=Path(tempfile.mkdtemp(prefix="figure_sources_"))
    try:
        (td/"data").mkdir(); shutil.copy2(ROOT/"data"/"sources.csv",td/"data"/"sources.csv"); shutil.copy2(ROOT/"data"/"data_dictionary.csv",td/"data"/"data_dictionary.csv")
        m.build_publication_sources(ROOT/"results"/"reference",td)
        refs=sorted((ROOT/"figures"/"source_data").glob("*.csv")); assert len(refs)==13
        for p in refs: compare(td/"figures"/"source_data"/p.name,p)
        print("PUBLICATION_FIGURE_SOURCE_DATA_TEST_PASS figures=13 tolerance=1e-12")
    finally: shutil.rmtree(td,ignore_errors=True)
if __name__=="__main__": main()
