import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Glyph.*missing from font")

GREEN = "#1DB954"
PALETTE = [
    "#1DB954", "#1ED760", "#169c46", "#0e6b31",
    "#535353", "#b3b3b3", "#f573a0", "#7c4dff",
    "#ff8a3d", "#2196f3", "#ffc107", "#00bcd4",
]

mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

CYCLE = plt.cycler(color=PALETTE)
mpl.rcParams["axes.prop_cycle"] = CYCLE


def finish(fig, out_path, title=None):
    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)
