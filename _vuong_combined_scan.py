"""
Orchestrator: scan a (f_i_mult, k_wall) 2D grid, single shared k_wall applied
to both phases, one fresh subprocess per grid point, sequentially. Skips
points already cached. Mirrors _vuong_fionly_scan.py's architecture.
"""
import subprocess, sys, os, itertools
GRID = [0.6, 0.9, 1.3, 1.8, 2.4]
for f, k in itertools.product(GRID, GRID):
    if os.path.exists(f'combined_cache/f_{f:.4f}_k_{k:.4f}.pkl'):
        print('cached', f, k, flush=True); continue
    print('running', f, k, flush=True)
    subprocess.run([sys.executable, '_vuong_combined_worker.py', str(f), str(k)],
                    check=True, timeout=1500)
