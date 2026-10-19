"""RVE thumbnails used inside the Word-native Fig. 1 (paper/flowchart.py): the same geometry, seeds and grey levels
as scripts/figs/fig01_framework.py. Output: paper/figs/fig01_rve_off.png, paper/figs/fig01_rve_on.png"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.dirname(__file__))
from mfhomog import rve
from common import FIGS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for name, (vf, gap, seed) in {'off': (0.55, 0.08, 11), 'on': (0.40, 0.15, 23)}.items():
    Xc, r = rve.generate(vf, 36, gap, 400, seed=seed)
    img = rve.rasterise(Xc, r, 96)
    fig = plt.figure(figsize=(1.0, 1.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(img.T, origin='lower', cmap='Greys', vmin=-0.4, vmax=1.3, interpolation='nearest')
    fig.savefig(os.path.join(FIGS, f'fig01_rve_{name}.png'), dpi=600); plt.close(fig)
print('saved')
