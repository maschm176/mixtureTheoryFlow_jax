"""
Orchestrator: scan k_wall (interfacial friction fixed at f_i_mult=1) over a
grid, one fresh subprocess per grid point, sequentially. Skips points already
cached. Mirrors _vuong_fionly_scan.py.
"""
import subprocess, sys, os
GRID = [float(x) for x in sys.argv[1:]] or [0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 2.5]
for k in GRID:
    if os.path.exists(f'wallonly_cache/k_{k:.4f}.pkl'):
        print('cached', k, flush=True); continue
    print('running', k, flush=True)
    subprocess.run([sys.executable, '_vuong_wallonly_worker.py', str(k)], check=True, timeout=1500)
