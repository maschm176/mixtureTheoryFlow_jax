"""
Orchestrator: scan f_i_mult (wall friction fixed at 1) over a grid, one fresh
subprocess per grid point, sequentially. Skips points already cached.
"""
import subprocess, sys, os
GRID = [float(x) for x in sys.argv[1:]] or [0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 2.5]
for f in GRID:
    if os.path.exists(f'fionly_cache/f_{f:.4f}.pkl'):
        print('cached', f, flush=True); continue
    print('running', f, flush=True)
    subprocess.run([sys.executable, '_vuong_fionly_worker.py', str(f)], check=True, timeout=1500)
