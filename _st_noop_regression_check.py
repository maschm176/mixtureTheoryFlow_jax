"""
Regression check for the new joint-implicit surface-tension half-step:
at sigma=0.0 (the default), advance_state/run_real_data_condition must
reproduce the pre-change solver's results bit-identically (or to
floating-point noise) -- same methodology used to validate the earlier
_bisect_ht multi-root fix (git-stash comparison of old vs. new solver).

Loads the OLD (pre-ST-change) solver module via `git show HEAD:...` into
a separate module name, and the NEW (working-tree) solver as normal,
then runs the same representative real-data condition through
run_real_data_condition on both and diffs every returned field.
"""
import importlib.util
import subprocess
import sys

import jax
jax.config.update("jax_enable_x64", True)

OLD_SRC = subprocess.run(
    ["git", "show", "HEAD:mixtureTheoryFlow_jax_solver_water_gas.py"],
    cwd="/Users/mschmitz/Documents/MixtureTheoryCode/mixtureTheoryFlow_jax",
    capture_output=True, text=True, check=True,
).stdout

OLD_PATH = "/tmp/_old_solver_water_gas.py"
with open(OLD_PATH, "w") as f:
    f.write(OLD_SRC)

spec = importlib.util.spec_from_file_location("solver_old", OLD_PATH)
solver_old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver_old)

sys.path.insert(0, "/Users/mschmitz/Documents/MixtureTheoryCode/mixtureTheoryFlow_jax")
import mixtureTheoryFlow_jax_solver_water_gas as solver_new

# A representative Vuong-like high-holdup-ish condition plus a plain
# low-holdup Fan-like condition -- want the no-op check to hold broadly,
# not just on one easy case.
CONDITIONS = [
    dict(U_sl=0.10030, U_sg=5.503, D=0.156, rho_l=760.0, rho_g=24.2,
         mu_l=1.35e-3, mu_g=1.5e-5, L=85.0, N=60, n_transit_times=5.0),
    dict(U_sl=0.018551, U_sg=15.110, D=0.0508, rho_l=1000.0, rho_g=2.57,
         mu_l=1.0e-3, mu_g=1.8e-5, L=19.0, N=60, n_transit_times=5.0),
]

all_ok = True
for i, cond in enumerate(CONDITIONS):
    r_old = solver_old.run_real_data_condition(**cond)
    r_new = solver_new.run_real_data_condition(**cond, sigma=0.0)

    print(f"--- condition {i} ---")
    for k in r_old:
        v_old, v_new = r_old[k], r_new[k]
        if isinstance(v_old, float):
            diff = abs(v_old - v_new)
            ok = diff < 1e-10
        else:
            diff = None
            ok = v_old == v_new
        all_ok &= ok
        print(f"  {k:20s} old={v_old!r:>20}  new={v_new!r:>20}  diff={diff}  ok={ok}")

print()
print("ALL OK" if all_ok else "MISMATCH FOUND")
