#!/usr/bin/env python3
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def text_of(path):
    nb=json.loads(path.read_text(encoding='utf-8'))
    text='\n'.join(''.join(c.get('source',[])) for c in nb.get('cells',[]))
    for c in nb.get('cells',[]):
        if c.get('cell_type')=='code':
            assert not c.get('outputs'),'Notebook contains saved outputs'
            assert c.get('execution_count') is None,'Notebook contains saved execution count'
    return text

def main():
    n1=text_of(ROOT/'notebooks/01_prepare_material_climate_analysis.ipynb')
    n2=text_of(ROOT/'notebooks/02_reproduce_material_climate_results.ipynb')
    for token in ['code/01_prepare_data.py','code/02_analyze_progress.py','code/05_build_extended_publication_evidence.py','tests/test_extended_publication_evidence.py','package-analysis']:
        assert token in n1,f'Notebook 1 missing contract token: {token}'
    required=['code/05_build_extended_publication_evidence.py','code/04_render_publication_figures.py','--output-root','work/publication','compare-analysis','compare-publication','package-publication','eu27-material-climate-publication-results.zip','receipt["git_commit"]','repository_manifest_sha256']
    for token in required: assert token in n2,f'Notebook 2 missing contract token: {token}'
    assert 'code/03_build_results.py' not in n2
    assert 'package-results' not in n2
    assert 'eu27-material-climate-results.zip' not in n2
    assert n2.count('files.download(')==1,'Notebook 2 must download exactly one final publication-results ZIP'
    print('NOTEBOOK_PUBLICATION_CONTRACT_TEST_PASS extended_evidence=both_notebooks publication_zip=single_download saved_outputs=0')
if __name__=='__main__': main()
