"""Numbers quoted in section 4.8 (external data sets), recomputed from the result files.
Inputs: results/ext_mnist.csv, results/mnist_published.json, results/ext_nf_final.csv, results/bounds_stats.json,
the two external repositories (paths as in run_external_*.py).
Usage: python3 -I scripts/check_sec48.py results/numbers_sec48.json"""
import sys, os, json, zipfile, io
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))


def r(x, k=3):
    return float(np.round(x, k))


def mnist():
    d = pd.read_csv(os.path.join(ROOT, 'results', 'ext_mnist.csv'))
    pub = json.load(open(os.path.join(ROOT, 'results', 'mnist_published.json')))
    z = zipfile.ZipFile(os.path.join(os.environ.get('EXT_DIR', 'external'), 'mm_Mechanical-MNIST-Transfer-Learning', 'sample_data.zip'))
    rd = lambda n: np.loadtxt(io.BytesIO(z.read(f'sample_data/{n}')))
    yH, yL = rd('UE_psi_test.txt'), rd('UE_CM_28_perturb_psi_test.txt')
    out = dict(n_train=int(len(rd('UE_psi_train.txt'))), n_test=int(len(yH)),
               LF_over_HF_median=float(np.median(yL / yH)), corr_log=r(np.corrcoef(np.log(yL), np.log(yH))[0, 1]),
               published=pub)
    C, L, G = 'Corrector GP (LF at query)', 'LF + log-linear map', 'GP (HF only)'
    for nH in [100, 1000]:
        s = d[d.nH == nH].pivot(index='rep', columns='method', values='mape')
        mx = d[d.nH == nH].pivot(index='rep', columns='method', values='maxape')
        o = dict(reps=int(len(s)), COR_mean=r(s[C].mean()), COR_sd=r(s[C].std()), COR_rep0=r(s.loc[0, C]),
                 LFmap_mean=r(s[L].mean()), LFmap_sd=r(s[L].std()),
                 COR_vs_LFmap_mean_reduction_pct=r((1 - s[C].mean() / s[L].mean()) * 100, 1),
                 COR_beats_LFmap=int((s[C] < s[L]).sum()),
                 COR_max_ape_range=[r(mx[C].min(), 2), r(mx[C].max(), 2)])
        if G in s:
            o.update(GP_mean=r(s[G].mean()), GP_sd=r(s[G].std()))
        if nH == 100:
            tl = pub['TL_CNN_nH100_LF1000_mape']
            o['COR_vs_TLCNN_reduction_pct_range'] = [r((1 - s[C].max() / tl) * 100, 1), r((1 - s[C].min() / tl) * 100, 1)]
        out[f'nH{nH}'] = o
    return out


def nf():
    import run_external_nf as R
    d = pd.read_csv(os.path.join(ROOT, 'results', 'ext_nf_final.csv'))
    m = R.prep(pd.read_csv(f'{R.ROOT}/dataset.csv'))
    Y = m[R.TGT].values; LO, UP, _ = R.physics(m); t = (Y - LO) / (UP - LO)
    tt, _ = R.prep_test()
    out = dict(n_main=int(len(m)), n_groups=int(m.grp.nunique()), n_test=int(len(tt)), n_test_rve=int(tt.sim.nunique()),
               t_median=dict(zip(R.TGT, [r(v) for v in np.median(t, 0)])), frac_outside_bounds=float(((t < 0) | (t > 1)).mean()),
               published_GP_2520=r(d[d.method == 'PUB-GP-2520'].mape_mean.iloc[0], 2),
               HT_mape=r(d[d.method == 'HT'].mape_mean.mean(), 2))
    med = d[d.method != 'PUB-GP-2520'].groupby(['n_groups', 'method']).mape_mean.median().unstack()
    out['median'] = {int(k): {c: r(v, 2) for c, v in row.items() if np.isfinite(v)} for k, row in med.iterrows()}
    mx = d.groupby(['n_groups', 'method']).mape_mean.max().unstack()
    out['max_rep_mean_ng6'] = dict(BN_GP=r(mx.loc[6, 'BN-GP'], 1), GP=float(mx.loc[6, 'GP']))
    pw = {}
    for ng in [6, 12, 24, 48, 96]:
        s = d[d.n_groups == ng].pivot(index='rep', columns='method', values='mape_mean')
        pw[ng] = {o: dict(BN_wins=int((s['BN-GP'] < s[o]).sum()), n=int(s[o].notna().sum()),
                          p=float(wilcoxon(s['BN-GP'], s[o]).pvalue))
                  for o in ['GP', 'HT-GP', 'MLP', 'XGB', 'POLY3', 'HT'] if o in s and s[o].notna().all()}
    out['paired_BN_GP'] = pw
    b = json.load(open(os.path.join(ROOT, 'results', 'bounds_stats.json')))['pool']
    out['UD_t_median'] = {k: r(b[k]['t_median']) for k in ['k23', 'G23', 'G12']}
    return out


if __name__ == '__main__':
    res = dict(mnist=mnist(), nf=nf())
    json.dump(res, open(sys.argv[1], 'w'), indent=1)
    print(json.dumps(res, indent=1))
