"""
Joint fit of (f_i_mult, k_wl, k_wg) against BOTH holdup and dPdz, on a
stratified train/test split of the 36 Vuong (2016) SW conditions.

Train/test split: 7 natural U_sg clusters in the data (gas rate swept at
~5-6 liquid rates each). One condition per cluster is held out as test,
cycling which liquid-rate rank is held out across clusters so the test
set still spans the full U_sl AND U_sg range rather than clustering at
one corner -- same deterministic clustering used interactively to design
this split, reproduced here so the script is self-contained.

Loss = mean(holdup_rel_err^2) + W_DPDZ*mean(dpdz_rel_err^2) over the 29
TRAINING conditions only (kh_total>0 conditions excluded, same policy as
every other fit in this project). Central finite-difference gradient in
log-space over all three parameters (7 forward evals/epoch: center + +-h
for each of f_i_mult, k_wl, k_wg), Adam with best-epoch tracking from the
start (not retrofitted after instability, learning from the old solver's
k1/k2 first-attempt instability). Test conditions are evaluated each
epoch for visibility only -- never used in the gradient.

EXECUTION MODEL -- one fresh subprocess per 29-condition batch, not one
long-lived process for the whole run. History, in order:
  1. Multiprocessing (ProcessPoolExecutor, up to 10 workers) was tried
     first and abandoned after three consecutive failures on this 16GB
     machine: 10 workers thrashed the system into severe swap exhaustion
     twice (even after adding n_steps bucketing to cut down on redundant
     JIT compiles), and a 4-worker config with periodic worker recycling
     then hung silently instead -- the pool died with zero exception,
     apparently because a replacement worker couldn't spawn promptly
     under sustained swap pressure.
  2. Reverting to plain serial (one single long-lived process, no
     bucketing, exactly the config the original 5-epoch smoke test used
     successfully) was tried next -- and ALSO failed, differently: a
     direct, isolated diagnostic (_diag_percall_timing.py) confirmed one
     clean batch of 29 conditions takes a flat ~230s with zero slowdown,
     but the SAME work embedded in the real fit script (which needs up
     to 7 such batches -- center + 6 finite-difference perturbations --
     within ONE process per epoch) consistently took 2-4x longer than
     the simple 7x extrapolation and climbed toward dangerous swap levels
     over ~90 minutes, with the process itself never hanging or crashing.
     Best explanation: n_steps differs for nearly every (condition,
     parameter) pair, so nearly every call is a fresh, large XLA compile;
     JAX never evicts a process's compile cache, and accumulating well
     beyond the ~29-distinct-compile scale that was directly proven clean
     apparently hits memory-allocator overhead (fragmentation from many
     large, never-freed buffers) that a 29-call diagnostic never reaches.
  3. Fix: tear the process down completely after every 29-condition batch
     and start fresh, via subprocess -- see _vuong_eval_worker.py, which
     does exactly the diagnostic's proven-clean job (one batch, one
     parameter point, exit) and nothing more. This is the same underlying
     fix as multiprocessing's max_tasks_per_child (bound how many
     distinct compiles one process can accumulate before being recycled)
     but applied strictly SEQUENTIALLY, with no persistent pool to hang
     if a respawn is slow -- each subprocess is launched, waited on
     synchronously, and collected before the next one starts.
This file's own top-level process (the "orchestrator") never calls
run_real_data_condition directly -- only eval_conditions_subprocess does,
and only inside a child process. The orchestrator's own memory footprint
stays whatever numpy/optax/pandas need, never accumulating JAX compile
state across epochs.
"""
import sys
import time
import pickle
import subprocess
import numpy as np
import pandas as pd
import optax

W_DPDZ = 1.0
H = 1e-3
LR = 0.06
N_EPOCHS = 30   # full run -- best-epoch tracking means an overshoot past the true optimum
                # (like the flat f_i_mult fit's own epoch 19->26 drift) doesn't lose the result
WORKER_TIMEOUT_S = 900   # generous safety net (a clean batch takes ~230s) -- catches a genuine
                         # hang rather than waiting on it forever


def build_split():
    df = pd.read_csv('vuong2016_sw_subset.csv').reset_index(drop=True)
    df = df.sort_values('vSG[m/s]').reset_index(drop=True)
    gaps = df['vSG[m/s]'].diff().fillna(999)
    df['cluster'] = (gaps > 0.1).cumsum()

    test_idx = []
    for i, (cid, grp) in enumerate(df.groupby('cluster')):
        grp_sorted = grp.sort_values('vSL[m/s]')
        rank_to_hold = i % len(grp_sorted)
        test_idx.append(grp_sorted.index[rank_to_hold])

    df['split'] = 'train'
    df.loc[test_idx, 'split'] = 'test'

    renamed = df.rename(columns={
        'vSL[m/s]': 'U_sl', 'vSG[m/s]': 'U_sg', 'rhoL[kg/m3]': 'rho_l', 'rhoG[kg/m3]': 'rho_g',
        'viscosityL[Pa.s]': 'mu_l', 'viscosityG[Pa.s]': 'mu_g', 'Pipe Diameter[m]': 'D',
        'Pipe Length[m]': 'L', 'HL[-]': 'psi', 'dP/dL[Pa/m]': 'dpdz_measured',
    })
    cols = ['U_sl', 'U_sg', 'D', 'rho_l', 'rho_g', 'mu_l', 'mu_g', 'L', 'psi', 'dpdz_measured']
    train = renamed.loc[renamed.split == 'train', cols].to_dict('records')
    test = renamed.loc[renamed.split == 'test', cols].to_dict('records')
    return train, test


def eval_conditions(f_i_mult, k_wl, k_wg, conditions):
    """Returns (loss_holdup, loss_dpdz, n_excluded, n_total). Only ever
    called from inside _vuong_eval_worker.py's own fresh process -- see
    module docstring. Imports run_real_data_condition locally so the
    orchestrator process (which imports THIS module for build_split)
    never pulls in jax itself."""
    from mixtureTheoryFlow_jax_solver_water_gas import run_real_data_condition

    alpha_sq, dpdz_sq = [], []
    n_excluded = 0
    for c in conditions:
        res = run_real_data_condition(c['U_sl'], c['U_sg'], c['D'], c['rho_l'], c['rho_g'],
                                       c['mu_l'], c['mu_g'], c['L'],
                                       f_i_mult=f_i_mult, k_wl=k_wl, k_wg=k_wg)
        if res['kh_total'] > 0:
            n_excluded += 1
            continue
        alpha_err = (res['alpha_l_pred'] - c['psi']) / c['psi']
        dpdz_err = (abs(res['dPdz_pred_liquid']) - c['dpdz_measured']) / c['dpdz_measured']
        alpha_sq.append(alpha_err ** 2)
        dpdz_sq.append(dpdz_err ** 2)
    loss_holdup = float(np.mean(alpha_sq)) if alpha_sq else float('nan')
    loss_dpdz = float(np.mean(dpdz_sq)) if dpdz_sq else float('nan')
    return loss_holdup, loss_dpdz, n_excluded, len(conditions)


def eval_conditions_subprocess(f_i_mult, k_wl, k_wg, split):
    """Runs _vuong_eval_worker.py in a brand-new process for this one
    (parameter point, split) batch, parses its one-line result, and lets
    the process exit -- see module docstring for why."""
    result = subprocess.run(
        [sys.executable, '_vuong_eval_worker.py',
         repr(float(f_i_mult)), repr(float(k_wl)), repr(float(k_wg)), split],
        capture_output=True, text=True, timeout=WORKER_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(f"worker failed (split={split}, f_i_mult={f_i_mult}, k_wl={k_wl}, "
                            f"k_wg={k_wg}):\n{result.stderr}")
    line = result.stdout.strip().splitlines()[-1]
    parts = dict(kv.split('=', 1) for kv in line.split())
    return (float(parts['loss_holdup']), float(parts['loss_dpdz']),
            int(parts['n_excluded']), int(parts['n_total']))


def combined_loss_subprocess(f_i_mult, k_wl, k_wg, split):
    lh, ld, n_exc, n_tot = eval_conditions_subprocess(f_i_mult, k_wl, k_wg, split)
    return lh + W_DPDZ * ld, lh, ld, n_exc, n_tot


def main():
    train, test = build_split()
    print(f"train: {len(train)} conditions, test: {len(test)} conditions")
    print("subprocess-per-batch execution (one fresh process per 29-condition "
          "evaluation -- see module docstring for why)")

    z = np.array([np.log(1.0), np.log(1.0), np.log(1.0)])  # [f_i_mult, k_wl, k_wg]
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(z)

    best_loss = np.inf
    best_epoch = None
    best_z = None

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        params_center = np.exp(z)
        loss_c, lh_c, ld_c, n_exc, n_tot = combined_loss_subprocess(*params_center, 'train')

        grad = np.zeros(3)
        for i in range(3):
            z_p = z.copy(); z_p[i] += H
            z_m = z.copy(); z_m[i] -= H
            loss_p, _, _, _, _ = combined_loss_subprocess(*np.exp(z_p), 'train')
            loss_m, _, _, _, _ = combined_loss_subprocess(*np.exp(z_m), 'train')
            grad[i] = (loss_p - loss_m) / (2 * H)

        updates, opt_state = optimizer.update(grad, opt_state)
        z = np.asarray(z + updates)   # optax returns jax arrays; keep z plain numpy for
                                       # the finite-difference in-place indexing above

        if loss_c < best_loss:
            best_loss = loss_c
            best_epoch = epoch
            best_z = params_center.copy()

        dt = time.time() - t0
        f_i_mult_c, k_wl_c, k_wg_c = params_center
        print(f"epoch {epoch:2d}  f_i_mult={f_i_mult_c:.4f}  k_wl={k_wl_c:.4f}  k_wg={k_wg_c:.4f}  "
              f"loss={loss_c:.6f} (holdup={lh_c:.6f} dpdz={ld_c:.6f})  "
              f"grad_z=[{grad[0]:+.4f},{grad[1]:+.4f},{grad[2]:+.4f}]  "
              f"n_excluded={n_exc}/{n_tot}  ({dt:.1f}s)", flush=True)

        with open('vuong2016_joint_fit_full.pkl', 'wb') as f:
            pickle.dump({
                'epoch': epoch, 'z': z, 'params': params_center,
                'loss': loss_c, 'loss_holdup': lh_c, 'loss_dpdz': ld_c,
                'best_epoch': best_epoch, 'best_z_params': best_z, 'best_loss': best_loss,
            }, f)

    print()
    print(f"Best epoch: {best_epoch}  f_i_mult={best_z[0]:.4f}  k_wl={best_z[1]:.4f}  k_wg={best_z[2]:.4f}  loss={best_loss:.6f}")

    print()
    print("=== Test-set evaluation at best-epoch params (NOT used in fitting) ===")
    lh_test, ld_test, n_exc_test, n_tot_test = eval_conditions_subprocess(*best_z, 'test')
    print(f"test holdup loss={lh_test:.6f}  test dPdz loss={ld_test:.6f}  n_excluded={n_exc_test}/{n_tot_test}")


if __name__ == '__main__':
    main()
