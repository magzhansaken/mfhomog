"""Figure style matching 机械工程学报 rules: text in figures 六号 (7.5 pt) 宋体 + Times New Roman,
axis labels '量/单位', inward ticks, curves distinguishable in black-and-white (line style + marker)
and in colour (CVD-validated categorical order). Column width 78 mm, full width 171 mm."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

COL_MM, FULL_MM = 78.0, 171.0
MM = 1 / 25.4
# categorical order validated for colour-vision deficiency (dataviz reference palette, light mode)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
GREY = '#8a8986'
LS = ['-', '--', '-.', ':', (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]
MK = ['o', 's', '^', 'v', 'D', 'x', '+', '*']


def setup():
    for f in ('/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf',
              '/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-italic.otf'):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass
    plt.rcParams.update({
        'font.family': ['TeX Gyre Termes', 'Noto Serif CJK SC'],
        'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'legend.fontsize': 7.0, 'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'lines.markersize': 3.5,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.top': True, 'ytick.right': True,
        'legend.frameon': False, 'savefig.dpi': 600, 'figure.dpi': 150, 'axes.unicode_minus': False,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'mathtext.fontset': 'custom', 'mathtext.rm': 'TeX Gyre Termes', 'mathtext.it': 'TeX Gyre Termes:italic',
        'mathtext.bf': 'TeX Gyre Termes:bold', 'mathtext.sf': 'TeX Gyre Termes', 'mathtext.tt': 'TeX Gyre Termes',
        'mathtext.cal': 'TeX Gyre Termes', 'mathtext.fallback': 'stix',
    })


def fig(width='col', h_mm=55, ncols=1, nrows=1, **kw):
    w = (COL_MM if width == 'col' else FULL_MM) * MM
    return plt.subplots(nrows, ncols, figsize=(w, h_mm * MM), **kw)


def panel_label(ax, text, y=-0.30):
    """Sub-figure label under the panel, e.g. '(a) 纵向剪切模量' (JME places it below each panel)."""
    ax.text(0.5, y, text, transform=ax.transAxes, ha='center', va='top', fontsize=7.5)


import re as _re
CJK_FONT = 'Noto Serif CJK SC'
_cjk = _re.compile(r'[\u3000-\u9fff\uff00-\uffef]')
_ascii_run = _re.compile(r"[A-Za-z0-9.,%:;/()\[\]+\-=~<>' ]+")


def M(s):
    """Prepare a label that mixes Chinese and mathtext. Matplotlib's mathtext has no font fallback, so the text
    object must use the CJK font as its base font and every Latin run outside $...$ is moved into \\mathrm{}
    (rendered in TeX Gyre Termes). Returns (string, dict(fontfamily=...)). Plain strings are returned unchanged."""
    if '$' not in s or not _cjk.search(s):
        return s, {}
    parts = s.split('$'); out = []
    for i, p in enumerate(parts):
        if i % 2:                       # math segment
            out.append('$' + p + '$')
        else:
            def rep(m):
                t = m.group(0)
                if not t.strip():
                    return t
                lead = len(t) - len(t.lstrip()); trail = len(t) - len(t.rstrip())
                core = t.strip().replace('%', r'\%').replace(' ', r'\ ')
                return ' ' * lead + r'$\mathrm{' + core + '}$' + ' ' * trail
            out.append(_ascii_run.sub(rep, p))
    return ''.join(out).replace('$$', ''), dict(fontfamily=CJK_FONT)


def xlabel(ax, s, **kw):
    t, f = M(s); ax.set_xlabel(t, **f, **kw)


def ylabel(ax, s, **kw):
    t, f = M(s); ax.set_ylabel(t, **f, **kw)


def text(ax, x, y, s, **kw):
    t, f = M(s); kw = {**f, **kw}; return ax.text(x, y, t, **kw)


def legend(ax, **kw):
    """Legend whose entries may mix Chinese and math: entries are converted with M() and the CJK font is used."""
    h, l = ax.get_legend_handles_labels()
    if any('$' in x and _cjk.search(x) for x in l):
        l = [M(x)[0] for x in l]; kw.setdefault('prop', {'family': CJK_FONT, 'size': 7.0})
    return ax.legend(h, l, **kw)


def logfmt(ax, axis='y', subs=(1.0, 2.0, 5.0)):
    """Log axis with plain decimal tick labels (0.2, 0.5, 1, 2, 5 ...) and no minor ticks."""
    from matplotlib.ticker import LogLocator, FuncFormatter, NullLocator
    a = ax.yaxis if axis == 'y' else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=subs)); a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
