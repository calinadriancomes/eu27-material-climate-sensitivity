#!/usr/bin/env python3
from pathlib import Path
from decimal import Decimal
from fractions import Fraction
import gzip, json, itertools, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from checks import RANK_DECIMALS

RAW=ROOT/"data"/"raw"/"cmur_env_ac_cur.json.gz"

def cat_codes(js,dim):
    cat=js["dimension"][dim]["category"]; idx=cat.get("index",{})
    if isinstance(idx,dict): return [k for k,v in sorted(idx.items(),key=lambda kv:kv[1])]
    if isinstance(idx,list): return list(idx)
    return list(cat.get("label",{}).keys())

def at(obj,i):
    if isinstance(obj,list): return obj[i] if i<len(obj) else None
    if isinstance(obj,dict): return obj.get(str(i),obj.get(i))
    return None

def exact_raw_series():
    # parse_float=Decimal preserves the exact finite decimal representation in the frozen JSON bytes
    js=json.loads(gzip.decompress(RAW.read_bytes()).decode("utf-8"),parse_float=Decimal)
    dims=list(js["id"]); codes=[cat_codes(js,d) for d in dims]; vals=js["value"]
    out={}
    for i,comb in enumerate(itertools.product(*codes)):
        rec=dict(zip(dims,comb)); geo=rec.get("geo"); t=rec.get("time")
        if geo and t and 2010<=int(t)<=2023:
            v=at(vals,i)
            if v is not None: out[(geo,int(t))]=Fraction(v)
    return out

def exact_ols(vals):
    xs=[Fraction(x,1) for x,y in vals]; ys=[y for x,y in vals]; n=Fraction(len(xs),1)
    num=n*sum((x*y for x,y in zip(xs,ys)),Fraction(0))-sum(xs,Fraction(0))*sum(ys,Fraction(0))
    den=n*sum((x*x for x in xs),Fraction(0))-sum(xs,Fraction(0))**2
    return num/den

def exact_theil_sen(vals):
    pts=[(Fraction(x,1),y) for x,y in vals]
    ss=sorted((y2-y1)/(x2-x1) for i,(x1,y1) in enumerate(pts) for x2,y2 in pts[i+1:])
    n=len(ss)
    return ss[n//2] if n%2 else (ss[n//2-1]+ss[n//2])/2

def calc(series,lo,hi,kind):
    geos=sorted({g for g,y in series})
    out={}
    for g in geos:
        vals=[(y,series[(g,y)]) for y in range(lo,hi+1)]
        out[g]=exact_ols(vals) if kind=="OLS" else exact_theil_sen(vals)
    return out

def assert_group(slopes, geos, expected):
    vals=[slopes[g] for g in geos]
    assert all(v==expected for v in vals), (geos,vals,expected)
    floats=[float(v) for v in vals]
    assert len({round(x,RANK_DECIMALS) for x in floats})==1

def main():
    s=exact_raw_series()
    early=calc(s,2010,2016,"OLS")
    assert_group(early,["PT","SK"],Fraction(1,14))
    assert_group(early,["BG","CZ"],Fraction(101,280))
    assert_group(early,["BE","NL"],Fraction(213,280))
    distinct=sorted(set(early.values()))
    min_gap=min(b-a for a,b in zip(distinct,distinct[1:]))
    assert min_gap==Fraction(1,140)
    # Round only before ranking so exact decimal ties are not split by floating-point noise.
    assert float(min_gap) > 1e8 * 10**(-RANK_DECIMALS)

    ets=calc(s,2010,2016,"TS")
    assert_group(ets,["CY","EE","HU","SK"],Fraction(1,10))
    lts=calc(s,2017,2023,"TS")
    assert_group(lts,["AT","IE"],Fraction(1,25))
    assert_group(lts,["BE","HR"],Fraction(1,4))

    print("EXACT_TIE_UNIT_TEST_PASS")
    print("OLS ties: PT=SK=1/14; BG=CZ=101/280; BE=NL=213/280")
    print("Minimum distinct exact EARLY OLS slope gap: 1/140")
    print("Theil-Sen ties: EARLY CY=EE=HU=SK=1/10; LATE AT=IE=1/25; BE=HR=1/4")
if __name__=="__main__": main()
