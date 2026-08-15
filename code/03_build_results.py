#!/usr/bin/env python3
from pathlib import Path
import argparse, json, shutil, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; sys.path.insert(0,str(HERE))
from checks import compare_analysis

matplotlib.rcParams["svg.hashsalt"]="eu27-material-climate-sensitivity"

def render(analysis, output):
    output=Path(output)
    if output.exists(): shutil.rmtree(output)
    for d in [output/"tables",output/"figures",output/"supplementary",output/"source_data"]: d.mkdir(parents=True,exist_ok=True)
    for sub in ["tables","supplementary","source_data"]:
        for p in sorted((analysis/sub).glob("*.csv")): shutil.copy2(p,output/sub/p.name)
    source=analysis/"source_data"
    d1=pd.read_csv(source/"full_representation_grid.csv",index_col=0); mat=d1.to_numpy(float)
    fig,ax=plt.subplots(figsize=(7.2,4.8)); im=ax.imshow(mat,aspect="auto")
    ax.set_xticks([0,1],["Territorial GHG","Consumption GHG"]); ax.set_yticks([0,1,2],["CMUR","DMC","RMC"])
    for i in range(3):
        for j in range(2): ax.text(j,i,f"{mat[i,j]:.3f}",ha="center",va="center")
    ax.set_title("2010–2023 progress-rank concordance (Spearman ρ)"); fig.colorbar(im,ax=ax,label="Spearman ρ"); fig.tight_layout()
    fig.savefig(output/"figures/full_representation_grid.png",dpi=220,metadata={"Date":None}); fig.savefig(output/"figures/full_representation_grid.svg",metadata={"Date":None}); plt.close(fig)

    d2=pd.read_csv(source/"country_rank_displacement.csv"); x=np.arange(len(d2)); width=0.25
    fig,ax=plt.subplots(figsize=(12,5.8)); ax.bar(x-width,d2.iloc[:,1],width,label="CMUR → DMC"); ax.bar(x,d2.iloc[:,2],width,label="DMC → RMC"); ax.bar(x+width,d2.iloc[:,3],width,label="Territorial → consumption GHG")
    ax.set_xticks(x,d2.geo,rotation=90); ax.set_ylabel("Absolute rank displacement"); ax.set_title("2010–2023 country-level rank displacement"); ax.legend(); fig.tight_layout()
    fig.savefig(output/"figures/country_rank_displacement.png",dpi=220,metadata={"Date":None}); fig.savefig(output/"figures/country_rank_displacement.svg",metadata={"Date":None}); plt.close(fig)

    d3=pd.read_csv(source/"temporal_contrasts.csv"); labels=["2010–2023","2010–2016","2017–2023"]; x=np.arange(3); width=0.18
    fig,ax=plt.subplots(figsize=(9,5.5))
    for j,(col,label) in enumerate([("D1_b_minus_a","D1"),("D2_c_minus_b","D2"),("D3_f_minus_c","D3"),("D_TOTAL_f_minus_a","D_TOTAL")]): ax.bar(x+(j-1.5)*width,d3[col],width,label=label)
    ax.axhline(0,linewidth=0.8); ax.set_xticks(x,labels); ax.set_ylabel("Difference in Spearman ρ"); ax.set_title("Representation contrasts by analysis period"); ax.legend(); fig.tight_layout()
    fig.savefig(output/"figures/temporal_contrasts.png",dpi=220,metadata={"Date":None}); fig.savefig(output/"figures/temporal_contrasts.svg",metadata={"Date":None}); plt.close(fig)
    validation={"status":"PASS","figures":6,"numeric_recalculation_in_renderer":False}
    (output/"validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--analysis",default=str(ROOT/"work/analysis")); ap.add_argument("--output",default=str(ROOT/"work/results")); args=ap.parse_args()
    analysis=Path(args.analysis)
    maxdiff=compare_analysis(ROOT,analysis)
    render(analysis,Path(args.output))
    print(f"RESULT_BUILD_PASS max_source_difference={maxdiff:.17g}")
if __name__=="__main__": main()
