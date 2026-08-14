#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
EU27={"AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","EL","HU","IE","IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE"}
def main():
    d=pd.read_csv(ROOT/"MAP_JOIN_PARITY.csv")
    assert len(d)==27 and set(d.scientific_geo)==EU27 and d.scientific_geo.is_unique
    assert (d.matched_geometry_n==1).all() and (d.duplicate_matches_n==0).all() and (d.status=="PASS").all()
    assert (d.expected_eu27_n==27).all() and (d.matched_eu27_n==27).all() and (d.missing_eu27_n==0).all()
    assert (d.scientific_value_recomputed==False).all()
    el=d[d.scientific_geo=="EL"].iloc[0]; assert el.natural_earth_iso_a2=="GR" and "explicitly" in el.match_method
    fr=d[d.scientific_geo=="FR"].iloc[0]; assert fr.natural_earth_adm0_a3=="FRA" and fr.natural_earth_admin=="France"
    print("MAP_JOIN_PARITY_TEST_PASS expected=27 matched=27 missing=0 duplicates=0 EL_to_GR=PASS")
if __name__=="__main__": main()
