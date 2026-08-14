#!/usr/bin/env python3
from pathlib import Path
import hashlib, shutil, subprocess, sys, tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from checks import publication_relpaths

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    td=Path(tempfile.mkdtemp(prefix='publication_empty_root_')); out=td/'publication'
    try:
        assert not out.exists()
        subprocess.run([sys.executable,str(ROOT/'code/04_render_publication_figures.py'),'--analysis',str(ROOT/'results/reference'),'--output-root',str(out),'--natural-earth',str(ROOT/'cartography/source/ne_10m_admin_0_countries.zip')],check=True,cwd=ROOT)
        rels=publication_relpaths()
        for rel in rels:
            a=out/rel
            b=(ROOT/'results'/'reference'/rel) if (len(rel.parts)>=2 and rel.parts[0]=='supplementary_data') else (ROOT/rel)
            assert a.is_file(),f'missing {rel}'
            assert b.is_file(),f'reference missing {rel}'
            assert sha(a)==sha(b),f'publication parity mismatch: {rel}'
        print(f'PUBLICATION_EMPTY_OUTPUT_ROOT_TEST_PASS files={len(rels)} differences=0')
    finally: shutil.rmtree(td,ignore_errors=True)
if __name__=='__main__': main()
