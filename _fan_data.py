"""
Shared data-loading helper for the Fan (2005) interfacial-friction fits --
the Fan-specific counterpart of _vuong_fit_joint.py's build_split(), kept
in its own module (not vuong_plots/, not any _vuong_*.py file) so none of
the Vuong fitting/plotting code is touched or overwritten by this work.

Fan (2005), full subset (fan2005_sw_subset.csv, 60 rows spanning
Inclination angle[deg] in {-2,-1,0,1,2}). Originally scoped to the 13
horizontal (angle==0) rows only; extended (2026-11-13) to all 60 rows
after a direct run_real_data_condition check (_inclined_real_data_check.py,
N=150, n_transit_times=150) against the 47 inclined rows found ZERO NaN
and -- for the 38/47 that are kh_total==0 -- alpha_l_pred matching the
closed-form Taitel-Dukler equilibrium to 4-5 significant figures, the same
clean-settling signature already validated for the horizontal rows. The
other 9/47 inclined rows show the same large, saturated kh_total already
documented for 2 of the 13 horizontal rows (a shear-driven instability
trigger at high U_sg, not an inclination-specific effect -- see CLAUDE.md,
"Fan (2005) zero-refit run surfaces a second instability trigger") and are
excluded by the same binary kh_total>0 policy, even though today's check
also showed some of these settle to the correct equilibrium despite a
nonzero kh_total -- kept conservative (excluded) for now, consistent with
the project's existing binary-cutoff policy, not reopened here.

kh_total exclusion -- 11 total conditions (2 horizontal + 9 inclined),
identified by their unique vSL[m/s] value (no two of the 60 rows share a
near-duplicate vSL, confirmed directly, so U_sl-only matching is safe):
  - 2 horizontal (confirmed by _fan_kh_check.py, see that script's output):
    0.014319654, 0.010300073
  - 9 inclined (confirmed by _inclined_real_data_check.py, 2026-11-13,
    rows at idx 0,3,9,15,17,19,20,35,36 of fan2005_sw_subset.csv, all at
    high U_sg~15-26 m/s, both signs of inclination):
    0.024133972, 0.018293584, 0.014227408, 0.010301941, 0.010222912,
    0.007958853, 0.007770405, 0.004022857, 0.003899203

SIGN CONVENTION, not independently verified against the Fan (2005) paper's
own text: assumed positive 'Inclination angle[deg]' = uphill, matching the
solver's own beta convention (module KNOWN LIMITATION section: "uphill
raises [holdup]...downhill lowers it") and the standard literature
convention (Taitel-Dukler, Beggs & Brill). beta_rad = radians(angle_deg).

Unlike Vuong, many of the kh-clean rows (13/38 inclined, 3/11 horizontal)
have no measured holdup (HL[-] == '-' in the source CSV) -- only dPdz. psi
is set to None for those; loss functions in the fitting scripts must mask
holdup terms on a per-condition basis rather than assuming every condition
has both targets.
"""
import csv
import math
from collections import defaultdict

CSV_PATH = 'fan2005_sw_subset.csv'

EXCLUDE_KH_USL = {
    0.014319654, 0.010300073,                                      # horizontal
    0.024133972, 0.018293584, 0.014227408, 0.010301941, 0.010222912,
    0.007958853, 0.007770405, 0.004022857, 0.003899203,             # inclined
}


def load_all_conditions(exclude_kh=True):
    conditions = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            U_sl = float(row['vSL[m/s]'])
            if exclude_kh and any(abs(U_sl - x) < 1e-9 for x in EXCLUDE_KH_USL):
                continue
            hl_raw = row['HL[-]'].strip()
            psi = float(hl_raw) if hl_raw not in ('', '-') else None
            angle_deg = float(row['Inclination angle[°]'])
            conditions.append({
                'U_sl': U_sl,
                'U_sg': float(row['vSG[m/s]']),
                'D': float(row['Pipe Diameter[m]']),
                'L': float(row['Pipe Length[m]']),
                'rho_l': float(row['rhoL[kg/m3]']),
                'rho_g': float(row['rhoG[kg/m3]']),
                'mu_l': float(row['viscosityL[Pa.s]']),
                'mu_g': float(row['viscosityG[Pa.s]']),
                'sigma': float(row['Surface Tension[N/m]']),
                'angle_deg': angle_deg,
                'beta': math.radians(angle_deg),
                'psi': psi,
                'dpdz_measured': float(row['dP/dL[Pa/m]']),
            })
    return conditions


def load_horizontal_conditions(exclude_kh=True):
    """Backward-compatible horizontal-only (angle==0) subset."""
    return [c for c in load_all_conditions(exclude_kh=exclude_kh) if c['angle_deg'] == 0.0]


def build_split(conditions):
    """
    Deterministic ~80/20 train/test split, grouping by (angle_deg, U_sg
    rounded to the nearest 5 m/s) -- Fan's natural "nominal U_sg level x
    inclination" clusters (U_sg sits near 15, 20, or 25 m/s throughout
    this dataset). One condition per cluster is held out as test, cycling
    which U_sl-rank is held out across clusters (same spirit as
    _vuong_fit_joint.py's build_split) so the test set spans the full
    U_sl/U_sg/angle range rather than clustering at one corner. The five
    U_sg~25 clusters have only one member each (one condition per angle
    at that gas rate) and stay entirely in train -- a 1-condition cluster
    can't produce a test example without losing that gas-rate level from
    train entirely.
    """
    groups = defaultdict(list)
    for c in conditions:
        key = (c['angle_deg'], round(c['U_sg'] / 5.0) * 5.0)
        groups[key].append(c)

    train, test = [], []
    for gi, key in enumerate(sorted(groups.keys())):
        members = sorted(groups[key], key=lambda c: -c['U_sl'])
        if len(members) < 2:
            train.extend(members)
            continue
        held_rank = gi % len(members)
        for i, c in enumerate(members):
            (test if i == held_rank else train).append(c)
    return train, test


if __name__ == '__main__':
    conds = load_all_conditions()
    n_with_psi = sum(1 for c in conds if c['psi'] is not None)
    print(f"n conditions (kh-clean) = {len(conds)}, of which {n_with_psi} have measured holdup")

    train, test = build_split(conds)
    n_train_psi = sum(1 for c in train if c['psi'] is not None)
    n_test_psi = sum(1 for c in test if c['psi'] is not None)
    print(f"train: n={len(train)} (with holdup={n_train_psi})")
    print(f"test:  n={len(test)} (with holdup={n_test_psi})")
    print("\ntest conditions (U_sl, U_sg, angle_deg):")
    for c in sorted(test, key=lambda c: (c['angle_deg'], -c['U_sg'])):
        psi_str = '--' if c['psi'] is None else f"{c['psi']:.4f}"
        print(f"  U_sl={c['U_sl']:.6f}  U_sg={c['U_sg']:7.3f}  angle={c['angle_deg']:5.1f}  psi={psi_str}")
