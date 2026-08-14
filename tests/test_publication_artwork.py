#!/usr/bin/env python3
from pathlib import Path
import re, subprocess
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
STEMS=[
"figure_1_primary_concordance_grid","figure_2_rank_concordance_geometry","figure_3_country_representation_consequences","figure_4_geography_progress_ranks","figure_5_temporal_representation_sensitivity","figure_6_deletion_robustness","figure_s1_temporal_concordance_matrices","figure_s2_geography_representation_shifts","figure_s3_geography_progress_scores","figure_s4_progress_score_geometry","figure_s5_broader_robustness_summary","figure_s6_sign_flip_overlap","figure_s7_country_discordance_matrix"]

def pdf_rasters(p):
    try:
        r=subprocess.run(["pdfimages","-list",str(p)],capture_output=True,text=True,check=True)
        lines=[x for x in r.stdout.splitlines() if re.match(r"\s*\d+\s+\d+\s+",x)]
        return len(lines)
    except FileNotFoundError:
        return 0 if b"/Subtype /Image" not in p.read_bytes() else 1

def main():
    for stem in STEMS:
        for ext in ["pdf","png","svg"]: assert (ROOT/"figures"/"reference"/f"{stem}.{ext}").is_file(),f"missing {stem}.{ext}"
        png=Image.open(ROOT/"figures"/"reference"/f"{stem}.png"); assert png.width>=700 and png.height>=400,(stem,png.size)
        svg=(ROOT/"figures"/"reference"/f"{stem}.svg").read_text(encoding="utf-8",errors="ignore"); assert "<image" not in svg.lower(),stem
        assert pdf_rasters(ROOT/"figures"/"reference"/f"{stem}.pdf")==0,stem
    print("PUBLICATION_ARTWORK_TEST_PASS figures=13 vector_pdf=PASS svg_embedded_raster=0")
if __name__=="__main__": main()
