"""
Standalone worker: evaluate (f_i_mult, k_wl, k_wg) against ONE split (train
or test) of the 29/7 Vuong conditions, print the result, exit.

Deliberately a fresh, short-lived process per invocation -- this is the
fix for the compile-cache-accumulation slowdown found during the real
30-epoch fit: a clean, isolated 29-condition batch (exactly this worker's
job) was directly measured at a flat ~230s with no slowdown (see
_diag_percall_timing.py), but the SAME work embedded in one long-lived
process handling all 7 per-epoch batches (203 calls total) consistently
took 2-4x longer and climbed toward dangerous swap levels, with no single
hung call -- consistent with JAX's never-evicted per-process compile
cache causing increasing memory-allocator overhead as MANY distinct
large XLA programs accumulate in one process's lifetime (n_steps differs
for nearly every (condition, parameter) pair, so each call is a fresh,
large compile). Tearing the process down completely after every batch
and starting fresh bounds this exactly at the proven-clean scale.

Usage: python3 _vuong_eval_worker.py <f_i_mult> <k_wl> <k_wg> <split>
Prints exactly one line: "loss_holdup=... loss_dpdz=... n_excluded=... n_total=..."
"""
import sys
from _vuong_fit_joint import build_split, eval_conditions

f_i_mult = float(sys.argv[1])
k_wl = float(sys.argv[2])
k_wg = float(sys.argv[3])
split = sys.argv[4]

train, test = build_split()
conditions = train if split == 'train' else test

lh, ld, n_exc, n_tot = eval_conditions(f_i_mult, k_wl, k_wg, conditions)
print(f"loss_holdup={lh!r} loss_dpdz={ld!r} n_excluded={n_exc} n_total={n_tot}")
