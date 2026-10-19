"""Numbers quoted in sections 1-3 of the paper that are not stored in another result file, recomputed from the data.
Usage: python3 -I scripts/check_sec1_3.py  -> prints and writes results/numbers_sec1_3.json
(Added in the pre-submission check, 17.10.2026; see results/PROTOCOL.md.)"""
import sys, os, json
import numpy as np
from scipy.stats import spearmanr
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from mfhomog import data as D, surrogates as S

out = {}
sets = {f: D.load(f'data/{f}.jsonl') for f in ['pool', 'lfx', 'val', 'test', 'scatter', 'ref']}
P = sets['pool']

# 3.1: cost of the data sets at the single-core benchmark rates of section 1.2 (results/cost_benchmark.json); the
# recorded times in data/*.jsonl are wall-clock times of 2 parallel worker processes on a loaded machine.
cb = json.load(open('results/cost_benchmark.json'))
med = {k: float(np.median(v)) for k, v in cb['raw'].items()}
bench, rec = {}, {}
for f in ['pool', 'lfx', 'val', 'test', 'scatter']:
    bench[f] = sum(med['geometry'] + sum(med[lev] for lev in r['levels']) for r in sets[f]) / 3600
    rec[f] = sum(r['t_geo'] + sum(sum(L['t']) for L in r['levels'].values()) for r in sets[f]) / 3600
out['data_cpu_h_benchmark_by_set'] = {k: round(v, 3) for k, v in bench.items()}
out['data_cpu_h_benchmark_5_modelling_sets'] = round(sum(bench.values()), 2)
out['data_wall_h_recorded_all_sets'] = round(sum(sum(r['t_geo'] + sum(sum(L['t']) for L in r['levels'].values())
                                                     for r in R) for R in sets.values()) / 3600, 2)
out['data_size_MB'] = round(sum(os.path.getsize('data/' + f) for f in os.listdir('data') if f.endswith('.jsonl')) / 1e6, 2)

# 1.1: arrangements with area fraction eta = Vf (1 + g)^2 > 0.6
eta = np.array([r['phys']['vf'] * (1 + r['phys']['gap']) ** 2 for r in P])
out['pool_eta_max'] = round(float(eta.max()), 4)
out['pool_share_eta_gt_0p6_pct'] = round(float((eta > 0.6).mean() * 100), 1)
op = json.load(open('results/order_parameter.json'))
out['rms_disp_over_nn_median_by_eta_bin'] = {k: v['rms_disp_over_nn'] for k, v in op['by_eta_bin'].items()}
out['psi6_global_median_by_eta_bin'] = {k: v['psi6_global'] for k, v in op['by_eta_bin'].items()}

# 1.1: residual anisotropy of the longitudinal shear tensor (median relative difference of the two diagonal entries)
for N in ['128', '256', '512']:
    Ga = np.array([r['levels'][N]['Ga'] for r in P]); a, b = Ga[:, 0, 0], Ga[:, 1, 1]
    out[f'shear_diag_rel_diff_median_pct_N{N}'] = round(float(np.median(np.abs(a - b) / ((a + b) / 2)) * 100), 2)

# 1.2: CG iterations, convergence, pixel volume fraction
its = {N: np.array([r['levels'][N]['its'] for r in P]) for N in ['64', '128', '256', '512']}
out['cg_its_median_antiplane_plane'] = {N: [float(np.median(v[:, 0])), float(np.median(v[:, 1]))] for N, v in its.items()}
out['cg_max_its_all_sets'] = int(max(max(L['its']) for R in sets.values() for r in R for L in r['levels'].values()))
out['cg_max_residual_all_sets'] = float(max(max(L['res']) for R in sets.values() for r in R for L in r['levels'].values()))
contrast = np.array([r['phys']['contrast'] for r in P])
out['spearman_plane_its_N512_vs_contrast'] = round(float(spearmanr(its['512'][:, 1], contrast)[0]), 3)
dv = np.array([abs(r['levels']['128']['vf_pix'] / r['phys']['vf'] - 1) * 100 for r in P])
out['vf_pixel_dev_N128_median_max_pct'] = [round(float(np.median(dv)), 3), round(float(dv.max()), 3)]

# 1.4: reference set and finite-element comparison
R = sets['ref']
out['ref_n'] = len(R)
out['ref_vf_range'] = [round(min(r['phys']['vf'] for r in R), 3), round(max(r['phys']['vf'] for r in R), 3)]
out['ref_contrast_range'] = [round(min(r['phys']['contrast'] for r in R), 2), round(max(r['phys']['contrast'] for r in R), 2)]
V = [json.loads(l) for l in open('results/verify_solver.jsonl')]
out['fe_n'] = len(V)
for N in ['64', '128', '256']:
    E = np.array([[r['levels'][N]['fft'][j] / r['levels'][N]['fe'][j] - 1 for j in range(3)] for r in V]) * 100
    out[f'fe_vs_fft_median_abs_rel_diff_pct_N{N}'] = np.round(np.median(np.abs(E), 0), 3).tolist()
    out[f'fft_below_fe_all_N{N}'] = bool((E < 0).all())
fb = json.load(open('results/fe_benchmark.json'))
out['fe_over_fft_time_ratio_median_N256'] = round(fb['summary']['256']['ratio_median'], 2)
out['fe_benchmark_n_rve'] = len(fb['raw']['256']['fe'])

# 2.1-2.3: coarse-grid error, modulus ranges, correction
X, Y, LO, UP = D.arrays(P, 'HF'); _, YL, _, _ = D.arrays(P, 128)
rel = (YL / Y - 1) * 100
out['coarse_N128_rel_err_median_pct_G12_G23_k23'] = np.round(np.median(rel, 0), 3).tolist()
out['coarse_N128_rel_err_max_pct'] = round(float(rel.max()), 2)
out['coarse_N128_n_not_stiffer_G12_G23_k23'] = (rel <= 0).sum(0).tolist()
out['moduli_max_over_min_G12_G23_k23'] = np.round(Y.max(0) / Y.min(0), 2).tolist()
out['G23_over_Gm_range'] = [round(float(Y[:, 1].min()), 3), round(float(Y[:, 1].max()), 3)]
d = S.to_s(Y, LO, UP, 'logit') - S.to_s(YL, LO, UP, 'logit')
out['correction_G23_range'] = [round(float(d[:, 1].min()), 3), round(float(d[:, 1].max()), 3)]
out['correction_G23_all_negative'] = bool((d[:, 1] < 0).all())
out['spearman_abs_correction_G23_vs_lgEf_Vf'] = [round(float(spearmanr(np.abs(d[:, 1]), X[:, 1])[0]), 3),
                                                round(float(spearmanr(np.abs(d[:, 1]), X[:, 0])[0]), 3)]
json.dump(out, open('results/numbers_sec1_3.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
