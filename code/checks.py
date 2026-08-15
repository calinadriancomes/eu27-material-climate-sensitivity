#!/usr/bin/env python3
from pathlib import Path
import argparse, csv, hashlib, json, os, re, shutil, sys, tempfile, zipfile
import numpy as np
import pandas as pd

EU27 = ["AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","EL","HU","IE","IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE"]
YEARS = list(range(2010, 2024))
PERIODS = {
    "FULL_2010_2023": list(range(2010, 2024)),
    "EARLY_2010_2016": list(range(2010, 2017)),
    "LATE_2017_2023": list(range(2017, 2024)),
}
EVEN_YEARS = list(range(2010, 2023, 2))
RANK_DECIMALS = 12
PANEL_SHA256 = "eb0b40eb58e2ea1e260d54c8d41504f35f61298e52501ce6d7b61b0f3bb6b317"

ROUTES = {
    "average_population_demo_gind": ("average_population_demo_gind.json.gz", "population", "population_status"),
    "cmur_env_ac_cur": ("cmur_env_ac_cur.json.gz", "cmur_pct", "cmur_status"),
    "cmur_sdg_12_41": ("cmur_sdg_12_41.json.gz", "cmur_sdg_pct", "cmur_sdg_status"),
    "dmc_total_env_ac_mfa": ("dmc_total_env_ac_mfa.json.gz", "dmc_total_ths_t", "dmc_total_status"),
    "dmc_per_capita_env_ac_mfa": ("dmc_per_capita_env_ac_mfa.json.gz", "dmc_pc_published_t", "dmc_pc_status"),
    "rmc_total_env_ac_rme": ("rmc_total_env_ac_rme.json.gz", "rmc_total_ths_t", "rmc_total_status"),
    "rmc_per_capita_env_ac_rme": ("rmc_per_capita_env_ac_rme.json.gz", "rmc_pc_published_t", "rmc_pc_status"),
    "territorial_ghg_total_sdg_13_10": ("territorial_ghg_total_sdg_13_10.json.gz", "terr_ghg_total_raw", "terr_total_status"),
    "territorial_ghg_per_capita_sdg_13_10": ("territorial_ghg_per_capita_sdg_13_10.json.gz", "terr_ghg_pc_published_raw", "terr_pc_status"),
    "consumption_ghg_total_cli_gge_foot": ("consumption_ghg_total_cli_gge_foot.json.gz", "cfoot_total_raw", "cfoot_total_status"),
    "consumption_ghg_per_capita_cli_gge_foot": ("consumption_ghg_per_capita_cli_gge_foot.json.gz", "cfoot_pc_published_raw", "cfoot_pc_status"),
    "production_ghg_total_cli_gge_foot": ("production_ghg_total_cli_gge_foot.json.gz", "cprod_total_raw", "cprod_status"),
}

REFERENCE_FILES = {
    "tables/main_results.csv": "tables/main_results.csv",
    "tables/representation_grid.csv": "tables/representation_grid.csv",
    "tables/country_rank_changes.csv": "tables/country_rank_changes.csv",
    "tables/country_rank_change_summary.csv": "tables/country_rank_change_summary.csv",
    "supplementary/accounting_bridge.csv": "supplementary/accounting_bridge.csv",
    "supplementary/rmc_status.csv": "supplementary/rmc_status.csv",
    "supplementary/robustness_summary.csv": "supplementary/robustness_summary.csv",
    "supplementary/tie_audit.csv": "supplementary/tie_audit.csv",
    "supplementary/progress_scores_all_specs.csv": "supplementary/progress_scores_all_specs.csv",
    "supplementary/country_results_all_specs.csv": "supplementary/country_results_all_specs.csv",
    "supplementary/published_vs_common_rank_summary.csv": "supplementary/published_vs_common_rank_summary.csv",
    "supplementary/published_vs_common_rank_detail.csv": "supplementary/published_vs_common_rank_detail.csv",
    "supplementary/leave_one_country_out.csv": "supplementary/leave_one_country_out.csv",
    "supplementary/leave_one_year_out.csv": "supplementary/leave_one_year_out.csv",
    "supplementary/cmur_even_year.csv": "supplementary/cmur_even_year.csv",
    "source_data/full_representation_grid.csv": "source_data/full_representation_grid.csv",
    "source_data/country_rank_displacement.csv": "source_data/country_rank_displacement.csv",
    "source_data/temporal_contrasts.csv": "source_data/temporal_contrasts.csv",
}

def sha256_file(path):
    path=Path(path); h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def verify_checksum_file(root, checksum_file):
    root=Path(root); checksum_file=Path(checksum_file)
    checked=0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        expected, rel=line.split(None,1); rel=rel.strip().lstrip("*")
        path=root/rel
        if not path.is_file(): raise RuntimeError(f"Missing file listed in {checksum_file.name}: {rel}")
        actual=sha256_file(path)
        if actual!=expected: raise RuntimeError(f"Checksum mismatch: {rel}")
        checked+=1
    return checked

def compare_csv(actual, expected, tol=1e-12):
    actual=Path(actual); expected=Path(expected)
    a=pd.read_csv(actual); e=pd.read_csv(expected)
    if list(a.columns)!=list(e.columns) or a.shape!=e.shape:
        raise RuntimeError(f"Schema or shape mismatch: {actual.name}")
    maxdiff=0.0
    for c in a.columns:
        av_num=pd.to_numeric(a[c],errors="coerce")
        ev_num=pd.to_numeric(e[c],errors="coerce")
        if av_num.notna().sum()==len(a) and ev_num.notna().sum()==len(e):
            av=av_num.to_numpy(float); ev=ev_num.to_numpy(float)
            d=float(np.max(np.abs(av-ev))) if len(av) else 0.0
            maxdiff=max(maxdiff,d)
            if d>tol: raise RuntimeError(f"Numeric mismatch in {actual.name}/{c}: {d} > {tol}")
        else:
            av=a[c].fillna("").astype(str).to_numpy(); ev=e[c].fillna("").astype(str).to_numpy()
            if not (av==ev).all(): raise RuntimeError(f"Text mismatch in {actual.name}/{c}")
    return maxdiff

def compare_panel(actual, reference, tol=1e-6):
    a=pd.read_csv(actual,keep_default_na=False).sort_values(["geo","year"]).reset_index(drop=True)
    e=pd.read_csv(reference,keep_default_na=False).sort_values(["geo","year"]).reset_index(drop=True)
    if set(a.columns)!=set(e.columns) or len(a)!=len(e): raise RuntimeError("Panel schema or row-count mismatch")
    a=a[list(e.columns)]
    maxdiff=0.0; maxcol=None
    for c in e.columns:
        if c=="geo" or c.endswith("_status"):
            if not (a[c].fillna("").astype(str).to_numpy()==e[c].fillna("").astype(str).to_numpy()).all():
                raise RuntimeError(f"Panel text/status mismatch: {c}")
        else:
            av=pd.to_numeric(a[c],errors="coerce").to_numpy(float); ev=pd.to_numeric(e[c],errors="coerce").to_numpy(float)
            d=float(np.nanmax(np.abs(av-ev)))
            if d>maxdiff: maxdiff,maxcol=d,c
    if maxdiff>tol: raise RuntimeError(f"Panel numeric mismatch: {maxdiff} in {maxcol}")
    return maxdiff,maxcol

def compare_analysis(repo_root, analysis_dir):
    repo_root=Path(repo_root); analysis_dir=Path(analysis_dir)
    ref=repo_root/"results"/"reference"
    maxdiff=0.0
    for rel in REFERENCE_FILES:
        d=compare_csv(analysis_dir/rel, ref/rel, 1e-12)
        maxdiff=max(maxdiff,d)
    return maxdiff

def safe_extract(zip_path, destination):
    zip_path=Path(zip_path); destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        bad=z.testzip()
        if bad is not None: raise RuntimeError(f"ZIP integrity check failed: {bad}")
        base=destination.resolve()
        for info in z.infolist():
            target=(destination/info.filename).resolve()
            if target!=base and base not in target.parents: raise RuntimeError("Unsafe path in ZIP")
        z.extractall(destination)

def write_checksums(root):
    root=Path(root); out=root/"checksums.sha256"
    if out.exists(): out.unlink()
    files=sorted(p for p in root.rglob("*") if p.is_file() and p.name!="checksums.sha256" and "__pycache__" not in p.relative_to(root).parts and ".pytest_cache" not in p.relative_to(root).parts and p.suffix!=".pyc" and "work" not in p.relative_to(root).parts)
    out.write_text("".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in files),encoding="utf-8")

def deterministic_zip(source_root, output_zip):
    source_root=Path(source_root); output_zip=Path(output_zip)
    if output_zip.exists(): output_zip.unlink()
    with zipfile.ZipFile(output_zip,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(x for x in source_root.rglob("*") if x.is_file()):
            rel=(Path(source_root.name)/p.relative_to(source_root)).as_posix()
            info=zipfile.ZipInfo(rel,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            z.writestr(info,p.read_bytes())
    with zipfile.ZipFile(output_zip) as z:
        if z.testzip() is not None: raise RuntimeError("ZIP integrity check failed")

def verify_repo(repo_root):
    repo_root=Path(repo_root)
    n=verify_checksum_file(repo_root,repo_root/"checksums.sha256")
    dn=verify_checksum_file(repo_root/"data",repo_root/"data"/"checksums.sha256")
    if sha256_file(repo_root/"data"/"panel_2010_2023.csv")!=PANEL_SHA256:
        raise RuntimeError("Reference panel checksum mismatch")
    return n,dn

def package_analysis(repo_root, output_zip, receipt_json=None, git_commit=None, repository_manifest_sha=None):
    repo_root=Path(repo_root); work=repo_root/"work"; analysis=work/"analysis"; data=work/"data"
    compare_analysis(repo_root,analysis)
    stage=work/"package_analysis"/"eu27-material-climate-analysis"
    if stage.parent.exists(): shutil.rmtree(stage.parent)
    (stage/"data"/"normalized").mkdir(parents=True)
    (stage/"numeric").mkdir(parents=True)
    shutil.copy2(data/"panel_2010_2023.csv",stage/"data"/"panel_2010_2023.csv")
    for p in sorted((data/"normalized").glob("*.csv")): shutil.copy2(p,stage/"data"/"normalized"/p.name)
    for sub in ["tables","supplementary","source_data","supplementary_data"]: shutil.copytree(analysis/sub,stage/"numeric"/sub)
    shutil.copy2(repo_root/"data"/"sources.csv",stage/"data"/"sources.csv")
    shutil.copy2(repo_root/"data"/"data_dictionary.csv",stage/"data"/"data_dictionary.csv")
    summary={"status":"PASS","rows":378,"countries":27,"years":"2010-2023","max_result_difference":compare_analysis(repo_root,analysis)}
    (stage/"validation.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (stage/"README.txt").write_text("Reproduced analysis package. The numeric directory contains baseline results, figure source data, and extended machine-readable publication evidence generated from the fixed Eurostat snapshot.\n",encoding="utf-8")
    if receipt_json and git_commit:
        raise RuntimeError("Use either a supplied receipt or a Git-bound public receipt, not both")
    if receipt_json:
        receipt=Path(receipt_json)
        if not receipt.is_file(): raise RuntimeError("Receipt file not found")
        json.loads(receipt.read_text(encoding="utf-8"))
        shutil.copy2(receipt,stage/"run_receipt.json")
    elif git_commit:
        manifest_sha=repository_manifest_sha or sha256_file(repo_root/"checksums.sha256")
        receipt={
            "schema_version":"1.0",
            "mode":"public_repository",
            "git_commit":str(git_commit),
            "repository_manifest_sha256":str(manifest_sha),
            "reference_panel_sha256":PANEL_SHA256,
        }
        (stage/"run_receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    write_checksums(stage); deterministic_zip(stage,output_zip)

def verify_external_package(zip_path, expected_root_name):
    tmp=Path(tempfile.mkdtemp(prefix="material_climate_"))
    try:
        safe_extract(zip_path,tmp); root=tmp/expected_root_name
        if not root.is_dir(): raise RuntimeError(f"Expected root folder not found: {expected_root_name}")
        n=verify_checksum_file(root,root/"checksums.sha256")
        return n
    finally: shutil.rmtree(tmp,ignore_errors=True)

def package_results(repo_root, results_dir, output_zip):
    repo_root=Path(repo_root); results_dir=Path(results_dir)
    stage=repo_root/"work"/"package_results"/"eu27-material-climate-results"
    if stage.parent.exists(): shutil.rmtree(stage.parent)
    stage.mkdir(parents=True)
    for sub in ["tables","figures","supplementary","source_data"]: shutil.copytree(results_dir/sub,stage/sub)
    if (results_dir/"validation.json").exists(): shutil.copy2(results_dir/"validation.json",stage/"validation.json")
    receipt=repo_root/"work"/"analysis_receipt.json"
    if receipt.exists():
        json.loads(receipt.read_text(encoding="utf-8"))
        shutil.copy2(receipt,stage/"run_receipt.json")
    (stage/"README.txt").write_text("Reproduced result package. Tables and supplementary files are numeric outputs from the analysis stage; figures are rendered only from the included source-data files.\n",encoding="utf-8")
    write_checksums(stage); deterministic_zip(stage,output_zip)



PUBLICATION_MAIN_TABLES = [
    "table_2_representation_contract_source.csv",
    "table_3_primary_concordance_and_contrasts.csv",
    "table_4_country_consequences.csv",
    "table_5_temporal_and_robustness.csv",
]
PUBLICATION_SUPPLEMENT_TABLES = [f"table_s{i}_{name}.csv" for i,name in [
    (1,"sources_units_provenance"),(2,"temporal_concordance_matrices"),(3,"companion_correlations"),(4,"temporal_ordered_contrasts"),
    (5,"extended_robustness"),(6,"temporal_bootstrap_diagnostics"),(7,"production_account_bridge"),(8,"rmc_provenance"),
    (9,"source_denominator_tie_scale_diagnostics")]]
PUBLICATION_FIGURE_STEMS = [
    "figure_1_primary_concordance_grid","figure_2_rank_concordance_geometry","figure_3_country_representation_consequences",
    "figure_4_geography_progress_ranks","figure_5_temporal_representation_sensitivity","figure_6_deletion_robustness",
    "figure_s1_temporal_concordance_matrices","figure_s2_geography_representation_shifts","figure_s3_geography_progress_scores",
    "figure_s4_progress_score_geometry","figure_s5_broader_robustness_summary","figure_s6_sign_flip_overlap","figure_s7_country_discordance_matrix",
]
PUBLICATION_SUPPLEMENTARY_DATA = [
    "symmetric_grid_and_temporal_effects.csv",
    "all_monotone_paths.csv",
    "within_axis_concordance.csv",
    "country_point_sign_and_loyo_stability.csv",
    "point_sign_disagreement_loyo_cases.csv",
    "point_sign_disagreement_loyo_summary.csv",
    "theil_sen_country_signs.csv",
    "theil_sen_disagreement_cases.csv",
    "theil_sen_disagreement_summary.csv",
    "trend_interval_diagnostics.csv",
    "trend_interval_counts.csv",
    "source_status_audit.csv",
    "rmc_country_method_registry.csv",
    "rmc_official_evidence_register.csv",
    "bootstrap_config.json",
    "bootstrap_grid_summary.csv",
    "bootstrap_endpoint_summary.csv",
    "bootstrap_country_rank_diagnostic_intervals.csv",
    "bootstrap_slope_favorable_frequencies.csv",
    "bootstrap_sign_disagreement_frequencies.csv",
    "bootstrap_rank_displacement_diagnostics.csv",
    "bootstrap_grid_extreme_identity_frequencies.csv",
    "bootstrap_full_grid_robustness.csv",
    "bootstrap_paired_contrast_summary.csv",
    "full_grid_fixed_sensitivities.csv",
    "full_grid_deletion_runs.csv",
    "rmc_provenance_sensitivity_n20.csv",
    "rmc_provenance_group_grids.csv",
    "population_flag_rank_displacement_summary.csv",
    "software_and_method.json",
]

def publication_relpaths():
    rels=[]
    rels += [Path("results/reference/tables")/x for x in PUBLICATION_MAIN_TABLES]
    rels += [Path("supplementary_tables")/x for x in PUBLICATION_SUPPLEMENT_TABLES]
    rels += [Path("supplementary_data")/x for x in PUBLICATION_SUPPLEMENTARY_DATA]
    rels += [Path("figures/source_data")/f"{x}.csv" for x in PUBLICATION_FIGURE_STEMS]
    rels += [Path("figures/reference")/f"{x}.{ext}" for x in PUBLICATION_FIGURE_STEMS for ext in ["pdf","png","svg"]]
    rels += [Path("MAP_JOIN_PARITY.csv")]
    return rels

PUBLICATION_NUMERIC_TOL = 1e-12
SVG_NUMERIC_TOL = 1e-5

_NUMERIC_TOKEN_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _text_equal_with_numeric_tolerance(actual, expected, tol=PUBLICATION_NUMERIC_TOL):
    """Compare text exactly except for machine-scale floating-point tokens."""
    actual=str(actual); expected=str(expected)
    a_parts=_NUMERIC_TOKEN_RE.split(actual); e_parts=_NUMERIC_TOKEN_RE.split(expected)
    a_nums=_NUMERIC_TOKEN_RE.findall(actual); e_nums=_NUMERIC_TOKEN_RE.findall(expected)
    if a_parts!=e_parts or len(a_nums)!=len(e_nums):
        return False
    for a,e in zip(a_nums,e_nums):
        try:
            if abs(float(a)-float(e))>tol:
                return False
        except ValueError:
            return False
    return True


def _compare_publication_csv(actual, expected, tol=PUBLICATION_NUMERIC_TOL):
    """Parsed schema/value comparison for publication CSVs.

    Numeric columns are compared at the repository's declared 1e-12 tolerance.
    Mixed text fields remain exact apart from embedded machine-scale numeric tokens
    (for example the raw floating-point spans reported by the tie audit).
    """
    a=pd.read_csv(actual,keep_default_na=False)
    e=pd.read_csv(expected,keep_default_na=False)
    if list(a.columns)!=list(e.columns) or a.shape!=e.shape:
        return False, "schema_or_shape_difference"
    for c in a.columns:
        av_num=pd.to_numeric(a[c],errors="coerce")
        ev_num=pd.to_numeric(e[c],errors="coerce")
        if av_num.notna().sum()==len(a) and ev_num.notna().sum()==len(e):
            av=av_num.to_numpy(float); ev=ev_num.to_numpy(float)
            if len(av) and float(np.max(np.abs(av-ev)))>tol:
                return False, f"numeric_difference:{c}"
        else:
            for av,ev in zip(a[c].astype(str),e[c].astype(str)):
                if not _text_equal_with_numeric_tolerance(av,ev,tol):
                    return False, f"text_difference:{c}"
    return True, ""


def _compare_png(actual, expected):
    """Require the reproduced PNG preview to be pixel-identical."""
    from PIL import Image
    with Image.open(actual) as a_img, Image.open(expected) as e_img:
        if a_img.mode!=e_img.mode or a_img.size!=e_img.size:
            return False, "png_geometry_difference"
        a=np.asarray(a_img); e=np.asarray(e_img)
        if a.shape!=e.shape or not np.array_equal(a,e):
            return False, "png_pixel_difference"
    return True, ""


def _compare_pdf_structure(actual, expected):
    """Check vector-PDF structure without requiring backend byte identity."""
    a=Path(actual).read_bytes(); e=Path(expected).read_bytes()
    if not a.startswith(b"%PDF") or not e.startswith(b"%PDF"):
        return False, "invalid_pdf"
    if b"/Subtype /Image" in a:
        return False, "unexpected_pdf_raster"
    page_re=re.compile(br"/Type\s*/Page(?!s)")
    if len(page_re.findall(a))!=len(page_re.findall(e)):
        return False, "pdf_page_structure_difference"
    return True, ""


def _compare_svg_structure(actual, expected, tol=SVG_NUMERIC_TOL):
    """Compare SVG structure while tolerating insignificant coordinate serialization."""
    import xml.etree.ElementTree as ET
    try:
        ar=ET.parse(actual).getroot(); er=ET.parse(expected).getroot()
    except ET.ParseError:
        return False, "invalid_svg"
    def signature(root):
        tags=[]; images=0
        for node in root.iter():
            tag=node.tag.split("}")[-1]
            tags.append(tag)
            if tag.lower()=="image": images+=1
        return tags,images
    a_tags,a_images=signature(ar); e_tags,e_images=signature(er)
    if a_images or e_images:
        return False, "unexpected_svg_raster"
    if a_tags!=e_tags:
        return False, "svg_structure_difference"
    # The public renderer is deterministic; when serialization differs, require
    # all non-numeric SVG text to remain identical and coordinates to differ by
    # no more than a tiny display-only tolerance.
    at=Path(actual).read_text(encoding="utf-8"); et=Path(expected).read_text(encoding="utf-8")
    if not _text_equal_with_numeric_tolerance(at,et,tol):
        return False, "svg_content_difference"
    return True, ""


def compare_publication(repo_root, publication_dir):
    repo_root=Path(repo_root); publication_dir=Path(publication_dir)
    mismatches=[]
    rels=publication_relpaths()
    for rel in rels:
        actual=publication_dir/rel
        # Extended supplementary evidence is stored once in the repository under
        # results/reference/supplementary_data. Reproduced publication packages
        # expose the same files under supplementary_data/.
        if len(rel.parts)>=2 and rel.parts[0]=="supplementary_data":
            expected=repo_root/"results"/"reference"/rel
        else:
            expected=repo_root/rel
        if not actual.is_file():
            mismatches.append((rel.as_posix(),"missing_reproduced_file")); continue
        if not expected.is_file():
            mismatches.append((rel.as_posix(),"missing_reference_file")); continue
        ext=rel.suffix.lower()
        if ext==".csv":
            ok,reason=_compare_publication_csv(actual,expected)
        elif ext==".png":
            ok,reason=_compare_png(actual,expected)
        elif ext==".pdf":
            ok,reason=_compare_pdf_structure(actual,expected)
        elif ext==".svg":
            ok,reason=_compare_svg_structure(actual,expected)
        else:
            ok,reason=(actual.read_bytes()==expected.read_bytes(),"byte_difference")
        if not ok:
            mismatches.append((rel.as_posix(),reason))
    return len(rels), mismatches

def package_publication(repo_root, publication_dir, output_zip):
    repo_root=Path(repo_root); publication_dir=Path(publication_dir); output_zip=Path(output_zip)
    n,mismatches=compare_publication(repo_root,publication_dir)
    if mismatches:
        raise RuntimeError(f"Publication output differs from checked-in reference files: {len(mismatches)} mismatches; first={mismatches[0]}")
    stage=repo_root/"work"/"package_publication"/"eu27-material-climate-publication-results"
    if stage.parent.exists(): shutil.rmtree(stage.parent)
    stage.mkdir(parents=True)
    for rel in publication_relpaths():
        dst=stage/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(publication_dir/rel,dst)
    ccm=publication_dir/"cartography"/"COUNTRY_CODE_MAP.csv"
    if ccm.is_file():
        dst=stage/"cartography"/"COUNTRY_CODE_MAP.csv"; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ccm,dst)
    receipt=repo_root/"work"/"analysis_receipt.json"
    if receipt.is_file():
        json.loads(receipt.read_text(encoding="utf-8")); shutil.copy2(receipt,stage/"run_receipt.json")
    validation={"status":"PASS","reference_publication_files_compared":n,"publication_output_differences":0,"main_table_sources":4,"supplement_table_sources":9,"supplementary_machine_readable_evidence_files":len(PUBLICATION_SUPPLEMENTARY_DATA),"figure_source_data":13,"figure_artwork_files":39,"map_join_parity_files":1}
    (stage/"validation.json").write_text(json.dumps(validation,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (stage/"README.txt").write_text("Reproduced final publication layer: manuscript table sources, Supplement table sources, machine-readable supplementary evidence, figure source data, publication artwork, and map-join parity output. All required files were validated against the checked-in reference publication files before packaging. Publication CSV values use the declared 1e-12 numerical tolerance with exact schema/text semantics; PNG previews are pixel-identical; vector PDF/SVG files are checked structurally, with only insignificant SVG coordinate serialization tolerated.\n",encoding="utf-8")
    write_checksums(stage); deterministic_zip(stage,output_zip)
    return n

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("verify"); p.add_argument("--root",default=".")
    p=sub.add_parser("compare-analysis"); p.add_argument("--root",default="."); p.add_argument("--analysis",default="work/analysis")
    p=sub.add_parser("package-analysis"); p.add_argument("--root",default="."); p.add_argument("--output",required=True); p.add_argument("--receipt-json"); p.add_argument("--git-commit"); p.add_argument("--repository-manifest-sha")
    p=sub.add_parser("verify-external"); p.add_argument("zip"); p.add_argument("root_name")
    p=sub.add_parser("package-results"); p.add_argument("--root",default="."); p.add_argument("--results",default="work/results"); p.add_argument("--output",required=True)
    p=sub.add_parser("compare-publication"); p.add_argument("--root",default="."); p.add_argument("--publication",default="work/publication")
    p=sub.add_parser("package-publication"); p.add_argument("--root",default="."); p.add_argument("--publication",default="work/publication"); p.add_argument("--output",required=True)
    args=ap.parse_args()
    if args.cmd=="verify":
        a,b=verify_repo(args.root); print(f"CHECKSUMS_PASS repository={a} data={b}")
    elif args.cmd=="compare-analysis": print(f"RESULT_COMPARISON_PASS max_difference={compare_analysis(args.root,Path(args.root)/args.analysis):.17g}")
    elif args.cmd=="package-analysis": package_analysis(args.root,args.output,args.receipt_json,args.git_commit,args.repository_manifest_sha); print(f"ANALYSIS_PACKAGE_READY {args.output}")
    elif args.cmd=="verify-external": print(f"PACKAGE_CHECKSUMS_PASS files={verify_external_package(args.zip,args.root_name)}")
    elif args.cmd=="package-results": package_results(args.root,Path(args.root)/args.results,args.output); print(f"RESULTS_PACKAGE_READY {args.output}")
    elif args.cmd=="compare-publication":
        n,mismatches=compare_publication(args.root,Path(args.root)/args.publication)
        if mismatches: raise RuntimeError(f"PUBLICATION_COMPARISON_FAIL differences={len(mismatches)} first={mismatches[0]}")
        print(f"PUBLICATION_COMPARISON_PASS files={n} differences=0")
    elif args.cmd=="package-publication":
        n=package_publication(args.root,Path(args.root)/args.publication,args.output)
        print(f"PUBLICATION_PACKAGE_READY files={n} differences=0 {args.output}")
if __name__=="__main__": main()
