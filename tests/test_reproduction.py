#!/usr/bin/env python3
from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"code"))
from checks import compare_panel, compare_analysis

def main():
    panel_diff,panel_col=compare_panel(ROOT/"work/data/panel_2010_2023.csv",ROOT/"data/panel_2010_2023.csv",1e-6)
    result_diff=compare_analysis(ROOT,ROOT/"work/analysis")
    out={"status":"PASS","max_panel_difference":panel_diff,"max_panel_difference_column":panel_col,"max_result_difference":result_diff}
    (ROOT/"work"/"reproduction_validation.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(f"REPRODUCTION_TEST_PASS max_result_difference={result_diff:.17g}")
if __name__=="__main__": main()
