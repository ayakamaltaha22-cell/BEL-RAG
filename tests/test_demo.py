import json,subprocess,sys,tempfile
from pathlib import Path
SCRIPT=Path(__file__).resolve().parents[1]/'src'/'demo_pipeline.py'
def test_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([sys.executable,str(SCRIPT),'--out',td],check=True,capture_output=True)
        m=json.loads((Path(td)/'demo_metrics.json').read_text())
        assert m['n_queries']==6
        assert m['policy_violations']==0
        assert 0<=m['exact_match']<=1
        assert (Path(td)/'demo_calibrator.json').exists()
if __name__=='__main__':
    test_end_to_end();print('PASS')
