"""Adapter for the Multi-Fidelity Residual Neural Process (MFRNP; Niu et al., ICML 2024; MIT licence,
github.com/Rose-STL-Lab/MFRNP). The original Supervisor/Model code is used unchanged; this file only writes our
data in its npz format, trains with its own validation-based model selection and reads the predictions back.
Changes to the defaults (documented in the paper): epochs 20000 instead of 50000, patience 4000 instead of 10000,
inputs standardised per feature before the library's scalar standardisation, outputs treated as a 1 x 3 'image'
(fid_lats = [1, 1], so the resizer is the identity), HF samples tiled to the LF count (lockstep loader)."""
import os, sys, types, tempfile, shutil, logging
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from .surrogates import to_s, from_s

import os as _os
MFRNP_DIR = _os.path.join(_os.environ.get('EXT_DIR', 'external'), 'mfrnp')   # scripts/get_external.sh


def _stub_ray():
    if 'ray' not in sys.modules:
        ray = types.ModuleType('ray'); tune = types.ModuleType('ray.tune'); train = types.ModuleType('ray.train')
        train.report = lambda *a, **k: None
        ray.tune, ray.train = tune, train
        sys.modules['ray'] = ray; sys.modules['ray.tune'] = tune; sys.modules['ray.train'] = train
    if 'wandb' not in sys.modules:
        sys.modules['wandb'] = types.ModuleType('wandb')


class MFRNP:
    def __init__(self, seed=0, epochs=20000, patience=4000):
        self.seed, self.epochs, self.patience = seed, epochs, patience
        self.name = 'MFRNP'

    def fit(self, XH, YH, LOH, UPH, XL=None, YL=None, LOL=None, UPL=None, **kw):
        self.data = (XH, YH, XL, YL); return self

    def predict(self, X, LO, UP, **kw):
        XH, YH, XL, YL = self.data
        _stub_ray(); torch.set_num_threads(1)
        if MFRNP_DIR not in sys.path:
            sys.path.insert(0, MFRNP_DIR)
        from model.supervisor import Supervisor
        logging.disable(logging.CRITICAL)
        tmp = tempfile.mkdtemp()
        rng = np.random.default_rng(self.seed)
        sx = StandardScaler().fit(np.vstack([XL, XH]))
        # The reference loader iterates all levels in lockstep (equal sizes) and takes the last 10 % as validation.
        # HF points are therefore tiled to the LF count, separately for the training (90 %) and validation (10 %)
        # parts, so that no HF point appears in both parts.
        nL = len(XL); nv = max(1, int(round(0.1 * nL))); nt = nL - nv
        iL = rng.permutation(nL)
        iH = rng.permutation(len(XH)); hv = max(1, int(round(0.1 * len(XH)))); Ht, Hv = iH[hv:], iH[:hv]
        idxH = np.concatenate([np.resize(Ht, nt), np.resize(Hv, nv)])
        np.savez(os.path.join(tmp, 'train_l1.npz'), x=sx.transform(XL)[iL], y=np.log(YL)[iL])
        np.savez(os.path.join(tmp, 'train_l2.npz'), x=sx.transform(XH)[idxH], y=np.log(YH)[idxH])
        dummy = np.zeros((len(X), YH.shape[1]))
        np.savez(os.path.join(tmp, 'test_l1.npz'), x=sx.transform(X), y=dummy)
        np.savez(os.path.join(tmp, 'test_l2.npz'), x=sx.transform(X), y=dummy)
        cfg = dict(base_dir=tmp, log_level='INFO', wandb_log=False,
                   data=dict(standard_transform=True, batch_size=2048, ablation=False, data_path=tmp,
                             save_dir=tmp, levels=2, random_seed=int(self.seed) + 1),
                   model=dict(hidden_dim=32, z_dim=32, context_percentage_low=0.2, context_percentage_high=0.25,
                              hidden_layers=2, l1_decay=0, device='cpu', fid_lats=[1, 1]),
                   train=dict(base_lr=1e-3, curr_epoch=0, epochs=self.epochs, epsilon=1e-3, lr_decay_ratio=0.85,
                              max_grad_norm=1, curr_patience=0, patience=self.patience, steps=[10000],
                              fidelity_weight=2, lower_fidelity_weight=1))
        cwd = os.getcwd(); os.chdir(tmp)
        try:
            sup = Supervisor(tmp, **cfg)
            sup.train()
            best = torch.load(os.path.join(tmp, 'best.pt'), weights_only=False)
            sup.model.load_state_dict(best['model_state_dict'])
            _, out = sup._run_epoch(sup.test_loader, train=False)
            pred = out['model_pred'].cpu().numpy()
            pred = sup.data['scaler_y'][-1].inverse_transform(pred)
        finally:
            os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)
        Y = np.exp(pred)
        return Y, Y, Y
