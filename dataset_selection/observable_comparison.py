import matplotlib.pyplot as plt
import pathlib
import numpy as np
import scipy
from matplotlib import patches, transforms, rc
from matplotlib.patches import Ellipse

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 14})
rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})

RESULT_DIR = pathlib.Path("../results/tcm_results")
USE_NORMALISED = False
EXPERIMENT = "CMS"
RESULT_DIR = pathlib.Path("../results/tcm_results")
DATE = "251021" if EXPERIMENT == "ATLAS" else "251022"

def get_suffix():
    return "-NORM" if USE_NORMALISED else ""

def load_central_value(path):
    with open(path) as f:
        line = f.readline()
    return np.fromstring(line.strip().strip("[]"), sep=" ")


def confidence_ellipse(ax, cov, mean, facecolor=None, confidence_level=95, **kwargs):
    eig_val, eig_vec = np.linalg.eigh(cov)
    order = np.argsort(eig_val)[::-1]
    eig_val, eig_vec = eig_val[order], eig_vec[:, order]

    chi2_qnt = scipy.stats.chi2.ppf(confidence_level / 100.0, 2)
    width, height = 2 * np.sqrt(chi2_qnt * eig_val)
    angle = np.degrees(np.arctan2(*eig_vec[:, 0][::-1]))

    ellipse = Ellipse(
        xy=mean, width=width, height=height, angle=angle,
        facecolor=facecolor, edgecolor=kwargs.get("edgecolor", "black"), linewidth=2
    )
    ellipse.set_facecolor((*plt.cm.colors.to_rgba(facecolor if facecolor else "C0")[:3], 0.15))
    ellipse.set_edgecolor((*plt.cm.colors.to_rgba(kwargs.get("edgecolor", "black"))[:3], 1.0))

    ax.add_patch(ellipse)
    ax.scatter(mean[0], mean[1], marker="x", color="black")
    ax.grid(True, which='major', linestyle='-', linewidth=0.8)
    ax.minorticks_on()
    from matplotlib.ticker import AutoMinorLocator
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))  # number of minor intervals per major tick
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(which='minor', linestyle=':', linewidth=0.6, alpha=0.7)
    return width, height

legend_elements = []
latex_dict = {"YT": r"$y_t$", "PTT": r"$p_T^t$", "MTTBAR": r"$m_{t\bar{t}}$", "YTTBAR": r"$y_{t\bar{t}}$",
                "MTTBAR-PTT": r"$(m_{t\bar{t}}, p_T^t)$", "MTTBAR-YTTBAR": r"$(m_{t\bar{t}}, y_{t\bar{t}})$"}

fig, axes = plt.subplots(3, 1,
                             figsize=(9 , 5 * 2),
                             squeeze=False)

def add_combined_analysis(ax, result_dir, experiment):

    date = "251021" if experiment == "ATLAS" else "251022"
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    combined_dirs = {
        f"YT{get_suffix()}": f"{date}-jth-dataset-selection-with-{experiment}_TTBAR_YT{get_suffix()}",
        f"PTT{get_suffix()}": f"{date}-jth-dataset-selection-with-{experiment}_TTBAR_PTT{get_suffix()}",
        f"MTTBAR{get_suffix()}": f"{date}-jth-dataset-selection-with-{experiment}_TTBAR_MTTBAR{get_suffix()}",
        f"YTTBAR{get_suffix()}": f"{date}-jth-dataset-selection-with-{experiment}_TTBAR_YTTBAR{get_suffix()}",
        f"MTTBAR-PTT{get_suffix()}": f"{date}-jth-dataset-selection-with-{experiment}_TTBAR_MTTBAR-PTT{get_suffix()}",
        f"MTTBAR-YTTBAR{get_suffix()}": f"{date}-jth-dataset-selection-with-{experiment}_TTBAR_MTTBAR-YTTBAR{get_suffix()}",
    }

    for i, (obs, dir_name) in enumerate(combined_dirs.items()):
        if obs == "MTTBAR-YTTBAR" and experiment == "ATLAS":
            dir_name = "251014-jth-dataset-selection-with-ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR"
        try:
            central_value = load_central_value(result_dir / dir_name / f"{dir_name}_central_value.dat")
            covmat = np.loadtxt(result_dir / dir_name / f"{dir_name}_covmat.dat")
        except FileNotFoundError:
            continue
        confidence_ellipse(ax, covmat, central_value, confidence_level=68, facecolor=colors[i], edgecolor=colors[i])
        if experiment == "ATLAS":

            face_rgba = (*plt.cm.colors.to_rgba(colors[i])[:3], 0.3)
            edge_rgba = (*plt.cm.colors.to_rgba(colors[i])[:3], 1.0)
            legend_elements.append(patches.Patch(facecolor=face_rgba, edgecolor=edge_rgba, label=latex_dict[obs]))


# get the unique legend elements


props = dict(facecolor='none', edgecolor='none')
for i, axi in enumerate(axes.flatten()):
    if i == 0:
        experiment = "ATLAS"
        exp_label = r"$\mathbf{ATLAS}$"
    elif i == 1:
        experiment = "CMS"
        exp_label = r"$\mathbf{CMS}$"
    else:
        experiment = "ATLAS_CMS"
        exp_label = r"$\mathbf{ATLAS+CMS}$"
        axi.set_xlabel(r"$m_t\;\mathrm{[GeV]}$")
    add_combined_analysis(axi, RESULT_DIR, experiment)
    axi.set_xlim(168, 180.5)
    axi.set_ylim(0.1175, 0.1225)

    axi.set_ylabel(r"$\alpha_s(m_Z)$")

    axi.text(0.03, 0.93, exp_label, transform=axi.transAxes, fontsize=17,
            verticalalignment='top', bbox=props, color='black')
    axi.text(0.97, 0.05, r"$\mathbf{NNLO}+\mathbf{MHOU}$", transform=axi.transAxes, fontsize=17,
            verticalalignment='bottom', bbox=props, color='black', horizontalalignment='right')


fig.legend(handles=legend_elements, loc="upper center", fontsize=17, frameon=False, bbox_to_anchor=(0.5, 0.98),
           ncol=3)




fig.savefig(f"observable_comparison{get_suffix()}.pdf")


