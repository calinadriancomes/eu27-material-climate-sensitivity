#!/usr/bin/env python3
from pathlib import Path
import runpy
import shutil
import tempfile

ROOT=Path(__file__).resolve().parents[1]
ns=runpy.run_path(str(ROOT/'code'/'checks.py'))
write_checksums=ns['write_checksums']
verify_checksum_file=ns['verify_checksum_file']

# Public prose/citation/legal metadata are Git-versioned but deliberately outside
# the computational checksum manifest. Computational inputs, code, notebooks,
# tests and canonical outputs remain covered.
manifest_paths=[]
for line in (ROOT/'checksums.sha256').read_text(encoding='utf-8').splitlines():
    if line.strip():
        _, rel=line.split(None,1)
        manifest_paths.append(rel.strip().lstrip('*'))

for rel in ['README.md','CITATION.cff','DATA.md','DATA_AVAILABILITY.md','THIRD_PARTY_DATA_AND_LICENSES.md','.gitignore','LICENSE']:
    assert rel not in manifest_paths, rel

for rel in ['code/checks.py','code/02_analyze_progress.py','notebooks/01_prepare_material_climate_analysis.ipynb','requirements-lock.txt','results/reference/tables/table_3_primary_concordance_and_contrasts.csv']:
    assert rel in manifest_paths, rel

# A prose-only edit must not invalidate the computational manifest.
tmp=Path(tempfile.mkdtemp(prefix='checksum_scope_'))
try:
    for rel in ['checksums.sha256','README.md']:
        src=ROOT/rel; dst=tmp/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    # Materialize all files listed in the manifest so the verifier can run.
    for rel in manifest_paths:
        src=ROOT/rel; dst=tmp/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    (tmp/'README.md').write_text((tmp/'README.md').read_text(encoding='utf-8')+'\nDocumentation-only edit.\n',encoding='utf-8')
    verify_checksum_file(tmp,tmp/'checksums.sha256')
finally:
    shutil.rmtree(tmp,ignore_errors=True)

print(f'REPOSITORY_CHECKSUM_SCOPE_TEST_PASS covered={len(manifest_paths)} metadata_edit_safe=PASS')
