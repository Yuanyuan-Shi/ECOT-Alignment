"""Resume the exact suspended experiment processes; never signal reused PIDs."""
from pathlib import Path
import datetime
import json
import os
import signal
import subprocess
import sys

ROOT = Path('/home/exx/Projects/ECOT-Alignment')
path = ROOT / 'runs/overnight_pause.json'
state = json.loads(path.read_text())
if state.get('resumed_at'):
    print('Already resumed')
    sys.exit(0)
if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != state['boot_id']:
    raise RuntimeError('Computer rebooted; use the saved handoff to restart unfinished work')
for group in state['groups']:
    pid = group['pgid']
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if stat[19] != group['start_ticks'] or group['controller'] not in command or os.getpgid(pid) != pid:
        raise RuntimeError(f'Process identity changed for {pid}; refusing to signal')
if '--check' in sys.argv:
    print('Both paused experiment groups verified; no signals sent')
    sys.exit(0)
for group in state['groups']:
    os.killpg(group['pgid'], signal.SIGCONT)
state['resumed_at'] = datetime.datetime.now().astimezone().isoformat()
tmp = path.with_suffix('.tmp')
tmp.write_text(json.dumps(state,indent=2)+'\n')
tmp.replace(path)
print('Resumed both experiment queues at',state['resumed_at'])
subprocess.run(['/home/exx/.conda/envs/openvla/bin/python','-m',
                'experiments.robot.libero.round3_move_comparison'],cwd=ROOT,check=True)
