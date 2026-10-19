"""External dataset 3: Mechanical MNIST (Lejeune 2020; data CC BY-SA 4.0 per OpenBU hdl.handle.net/2144/38693, the
GitHub README of the original repository states CC BY-SA 3.0 for the MNIST-derived subset; code MIT).
High fidelity: UE (uniaxial extension, fine mesh) strain energy; low fidelity: UE-CM-28-perturb (coarse 28x28 mesh,
small perturbation). Same budgets as Lejeune & Zhao (2021): 100 HF (+1000 LF) and 1000 HF; test = their 10000 test set.
Inputs: 784-pixel bitmaps -> PCA features fitted on the training images only.
Repeats: 10 at 100 HF, 3 at 1000 HF; rep 0 uses Lejeune & Zhao's subset (first n training examples).
Methods: LF raw, GP on HF only (100 HF only), corrector GP with the coarse solve of the query (log coordinate).
Usage: python3 scripts/run_external_mnist.py OUT.csv"""
import sys, os, time, zipfile, io
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mfhomog import surrogates as S

REPO = os.path.join(os.environ.get('EXT_DIR', 'external'), 'mm_Mechanical-MNIST-Transfer-Learning')   # scripts/get_external.sh


def load():
    z = zipfile.ZipFile(f'{REPO}/sample_data.zip')
    rd = lambda n: np.loadtxt(io.BytesIO(z.read(f'sample_data/{n}')))
    return (rd('mnist_img_train.txt'), rd('mnist_img_test.txt'), rd('UE_psi_train.txt'), rd('UE_psi_test.txt'),
            rd('UE_CM_28_perturb_psi_train.txt'), rd('UE_CM_28_perturb_psi_test.txt'))


def run(F, Ftr, Fte, yH, yL, yLte, iH, seed):
    """F: PCA features of all training images, Fte: of the test images (PCA fitted on the 60000 training bitmaps,
    unsupervised, no test image used)."""
    XH, XE = Ftr[iH], Fte
    one = lambda n: np.ones((n, 1))
    out = {'LF raw (coarse solve at query)': yLte.copy()}
    # LF is a small-perturbation coarse-mesh energy (different magnitude): calibrated LF baseline = log-linear map
    # fitted on the n_H training pairs (added 00:47 after the first repeat, see PROTOCOL.md)
    b, a = np.polyfit(np.log(yL[iH]), np.log(yH[iH]), 1)
    out['LF + log-linear map'] = np.exp(a + b * np.log(yLte))
    if len(iH) <= 100:   # single-fidelity GP only at the small budget (32-D GP on 1000 points is too costly here)
        g = S.GP('log', seed).fit(XH, yH[iH, None], one(len(iH)), one(len(iH)))
        out['GP (HF only)'] = g.predict(XE, one(len(XE)), one(len(XE)))[0][:, 0]
    # solver-in-the-loop corrector (log coordinate; no bounds are available for the strain energy)
    Z = np.column_stack([XH, np.log(yL[iH])]); ZE = np.column_stack([XE, np.log(yLte)])
    sx = StandardScaler().fit(Z)
    gc = S._sk_gp(Z.shape[1], seed).fit(sx.transform(Z), np.log(yH[iH]) - np.log(yL[iH]))
    out['Corrector GP (LF at query)'] = np.exp(np.log(yLte) + gc.predict(sx.transform(ZE)))
    return out


def main():
    outp = sys.argv[1]
    Itr, Ite, yH, yHte, yL, yLte = load()
    pca = PCA(32, random_state=0).fit(Itr)
    Ftr, Fte = pca.transform(Itr), pca.transform(Ite)
    for nH, reps in [(100, 10), (1000, 3)]:      # fixed before any result (compute time)
        for rep in range(reps):
            if rep == 0:      # exactly Lejeune & Zhao's subset: first nH training examples
                iH = np.arange(nH)
            else:
                iH = np.random.default_rng(777 + rep).choice(60000, nH, replace=False)
            t0 = time.time()
            res = run(None, Ftr, Fte, yH, yL, yLte, iH, rep)
            for k, p in res.items():
                e = np.abs(p / yHte - 1) * 100
                row = dict(method=k, nH=nH, rep=rep, mape=e.mean(), mae=np.abs(p - yHte).mean(), maxape=e.max(), t=time.time() - t0)
                pd.DataFrame([row]).to_csv(outp, mode='a', header=not os.path.exists(outp), index=False)
            print(nH, rep, {k: round(float((np.abs(v / yHte - 1) * 100).mean()), 2) for k, v in res.items()}, flush=True)


if __name__ == '__main__':
    main()
