"""Shared helpers for the result figures: load the final (or, for layout tests only, validation) results."""
import os, glob
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIGS = os.path.join(ROOT, 'paper', 'figs')

# display names and fixed style per method (colour follows the method in every figure)
STYLE = {
    'BN-COR': dict(label='本文混合模式', c=0, ls='-', mk='o'),
    'BN-AR1': dict(label='本文参数化模式', c=1, ls='--', mk='s'),
    'GP': dict(label='GP', c=2, ls='-.', mk='^'),
    'AR1': dict(label='AR1', c=3, ls=':', mk='v'),
    'NARGP': dict(label='NARGP', c=4, ls=(0, (5, 1, 1, 1)), mk='D'),
    'MFNN': dict(label='MFNN', c=5, ls=(0, (3, 1, 1, 1, 1, 1)), mk='x'),
    'BMLP': dict(label='BMLP', c=6, ls='--', mk='P'),
    'MFRNP': dict(label='MFRNP', c=7, ls='-.', mk='*'),
}
REF = {'LF': dict(label='LF ($N$=128)', ls='-.'), 'RICH': dict(label='RICH', ls=':'), 'MT': dict(label='MT', ls='--')}


def load(src='test'):
    if src == 'test':
        files = sorted(glob.glob(os.path.join(ROOT, 'results', 'test_[ABCD].csv')))
    else:   # validation files, used only to test figure layouts before the final run
        files = sorted(glob.glob(os.path.join(ROOT, 'results', 'val_*.csv')))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if src != 'test':
        d = d[(d.lf == 128) | d.method.isin(['RICH', 'GP', 'BN-GP'])]
    return d
