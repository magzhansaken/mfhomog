"""Test errors of the published Mechanical MNIST metamodels (Lejeune & Zhao 2021), computed from the prediction files
that the authors released in their repository (metamodels/*_test_predict.txt, *_test_correct.txt; MIT licence).
Also checks that their test targets are the same 10000 values (same order) as the test set used in
scripts/run_external_mnist.py (sample_data.zip, UE_psi_test.txt).
Usage: python3 -I scripts/mnist_published.py results/mnist_published.json"""
import sys, json, zipfile, io
import numpy as np

import os
REPO = os.path.join(os.environ.get('EXT_DIR', 'external'), 'mm_Mechanical-MNIST-Transfer-Learning')
M = f'{REPO}/metamodels'


def mape(p, y):
    return float(np.mean(np.abs(p / y - 1)) * 100)


def main():
    out = sys.argv[1]
    z = zipfile.ZipFile(f'{REPO}/sample_data.zip')
    y_ours = np.loadtxt(io.BytesIO(z.read('sample_data/UE_psi_test.txt')))
    c = np.loadtxt(f'{M}/LC_UE_test_correct.txt'); p = np.loadtxt(f'{M}/LC_UE_test_predict.txt')
    ct = np.loadtxt(f'{M}/TL_LC_UE_pre_train_UE_CM_1000_test_correct.txt')
    pt = np.loadtxt(f'{M}/TL_LC_UE_pre_train_UE_CM_1000_test_predict.txt')
    r = {
        'n_test': int(len(y_ours)),
        'targets_identical_max_rel_diff': float(np.max(np.abs(c[:, 0] / y_ours - 1))),
        'targets_identical_TL_max_rel_diff': float(np.max(np.abs(ct / y_ours - 1))),
        'CNN_nH100_mape': mape(p[:, 0], c[:, 0]),
        'CNN_nH1000_mape': mape(p[:, 1], c[:, 1]),
        'TL_CNN_nH100_LF1000_mape': mape(pt, ct),
        'CNN_nH100_maxape': float(np.max(np.abs(p[:, 0] / c[:, 0] - 1)) * 100),
        'TL_CNN_nH100_LF1000_maxape': float(np.max(np.abs(pt / ct - 1)) * 100),
        'CNN_nH1000_maxape': float(np.max(np.abs(p[:, 1] / c[:, 1] - 1)) * 100),
    }
    json.dump(r, open(out, 'w'), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == '__main__':
    main()
