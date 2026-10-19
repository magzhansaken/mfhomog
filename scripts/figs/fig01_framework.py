"""Fig. 1  Framework of the solver-in-the-loop bound-normalised hybrid model (schematic)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mfhomog import jmestyle as J, rve
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

J.setup()
Y0, H = 6.0, 74.0
fig = plt.figure(figsize=(J.FULL_MM * J.MM, H * J.MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 171); ax.set_ylim(Y0, Y0 + H); ax.axis('off')
EDGE = '#333333'


def box(x, y, w, h, text, fc='white', bold=False, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec=EDGE, lw=0.6, ls=ls))
    J.text(ax, x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7.5,
           fontweight='bold' if bold else 'normal', linespacing=1.3)


def arrow(x0, y0, x1, y1, style='-|>', ls='-'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7, lw=0.6, color=EDGE, ls=ls,
                                 shrinkA=0, shrinkB=0))


def rve_img(x, y, size, vf, gap, seed, N=96):
    Xc, r = rve.generate(vf, 36, gap, 400, seed=seed)
    img = rve.rasterise(Xc, r, N)
    a = fig.add_axes([x / 171, (y - Y0) / H, size / 171, size / H])
    a.imshow(img.T, origin='lower', cmap='Greys', vmin=-0.4, vmax=1.3, interpolation='nearest')
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_linewidth(0.5)


# stage labels and dividing line
ax.text(4, 76.0, '离线阶段', fontsize=7.5, fontweight='bold')
ax.text(4, 32.5, '在线阶段', fontsize=7.5, fontweight='bold')
ax.plot([2, 169], [36.0, 36.0], color=J.GREY, lw=0.5, ls=(0, (4, 2)))

# offline row
box(4, 52, 24, 18, '设计变量 $\\boldsymbol{x}$\n$V_\\mathrm{f}$, $\\lg(E_\\mathrm{f}/E_\\mathrm{m})$,\n$\\nu_\\mathrm{f}$, $\\nu_\\mathrm{m}$, $s_\\mathrm{g}$\nSobol 取样')
arrow(28.8, 61, 34.2, 61)
rve_img(35.5, 54.0, 13, 0.55, 0.08, 11)
ax.text(42, 51.2, '随机排布 RVE', ha='center', fontsize=7.5)
arrow(49.6, 64, 56.2, 66.5); arrow(49.6, 58, 56.2, 55.5)
box(57, 62, 33, 10, 'FFT 粗网格求解\n$N=128$ → $y_\\mathrm{L}$')
box(57, 47, 33, 12, 'FFT 高保真求解\n$N=256$, 512 → $y_\\mathrm{H}$\n(Richardson 外推)')
box(97, 57, 28, 12, 'HS 界限 $L$, $U$\n界限归一化坐标\n$s=\\mathrm{logit}[(y-L)/(U-L)]$', fc='#f2f2f2')
arrow(90.8, 67, 96.2, 65); arrow(90.8, 53, 96.2, 59.5)
box(132, 57, 35, 12, 'GP 学习校正量\n$\\delta(\\boldsymbol{x},s_\\mathrm{L})=s_\\mathrm{H}-s_\\mathrm{L}$\n($n_\\mathrm{H}=10\\sim160$)', fc='#f2f2f2', bold=False)
arrow(125.8, 63, 131.2, 63)
box(97, 40, 28, 11, '参数化模式\n(无在线求解)\n$s_\\mathrm{H}=\\rho m_\\mathrm{L}+c+\\delta_\\mathrm{p}$', ls='--')
arrow(111, 56.2, 111, 51.8, ls='--')
arrow(111, 39.2, 111, 25.6, ls='--')

# online row
rve_img(5, 12, 13, 0.40, 0.15, 23)
ax.text(11.5, 9.2, '待预测微结构', ha='center', fontsize=7.5)
arrow(18.8, 18.5, 25.2, 18.5)
box(26, 13, 27, 11, '一次粗网格求解\n$N=128$ → $s_\\mathrm{L}$')
arrow(53.8, 18.5, 59.2, 18.5)
box(60, 13, 34, 11, '$s_\\mathrm{H}=s_\\mathrm{L}+\\delta$\n$\\delta$ 取 GP 后验的 400 个样本')
arrow(94.8, 18.5, 100.2, 18.5)
box(101, 13, 30, 11, '逆变换\n$y=L+(U-L)/(1+\\mathrm{e}^{-s})$')
arrow(131.8, 18.5, 137.2, 18.5)
box(138, 9, 29, 19, '样本均值与 95% 区间\n$G_{12}$, $G_{23}$, $k_{23}$\n↓ Hill 精确关系\n$E_1$, $\\nu_{12}$, $E_2$, $\\nu_{23}$\n(满足 HS 界限)')
arrow(149.5, 56.2, 149.5, 31.0, style='-'); arrow(149.5, 31.0, 77, 31.0, style='-'); arrow(77, 31.0, 77, 24.8)
ax.text(132, 32.0, '训练好的校正 GP', ha='center', va='bottom', fontsize=7.5)

out = os.path.join(os.path.dirname(__file__), '..', '..', 'paper', 'figs', 'fig01_framework')
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
print('saved', out)
