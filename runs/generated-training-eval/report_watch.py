import json,subprocess,time
from pathlib import Path
ROOT=Path('/home/exx/Projects/ECOT-Alignment')
Q=ROOT/'runs/generated-training-eval'
while True:
    subprocess.run(['/home/exx/.conda/envs/openvla/bin/python',str(ROOT/'scripts/analysis/render_generated_training.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    s=json.loads((Q/'status.json').read_text())
    if s['status']!='running':break
    time.sleep(45)
