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
RESULT_DIR = pathlib.Path("../results/tcm_results")

if USE_NORMALISED:
    x_min, x_max = 168, 179
    y_min, y_max = 0.118, 0.1225
else:
    x_min, x_max = 168, 180.5
    y_min, y_max = 0.1175, 0.1225

def get_suffix():
    return "-NORM" if USE_NORMALISED else ""

colors_dict = {
    "YT": "C0",
    "PTT": "C1",
    "MTTBAR": "C2",
    "MTTBAR-cut": "C2",
    "YTTBAR": "C3",
    "MTTBAR-PTT": "C4",
    "MTTBAR-YTTBAR": "C5",
    "MTTBAR-YTTBAR-cut": "C5",
    "MTTBAR-YT": "C6",
    "PTT-YT": "C7",
}

# TODO: run ATLAS MTTBAR-PTT cut
combined_dirs = {
    "NNLO": {
        "ATLAS": {
            f"YT{get_suffix()}": f"251021-jth-dataset-selection-with-ATLAS_TTBAR_YT{get_suffix()}",
            f"PTT{get_suffix()}": f"251021-jth-dataset-selection-with-ATLAS_TTBAR_PTT{get_suffix()}",
            #f"MTTBAR{get_suffix()}": f"251021-jth-dataset-selection-with-ATLAS_TTBAR_MTTBAR{get_suffix()}",
            f"MTTBAR-cut": f"251105-jth-dataset-selection-with-ATLAS_TTBAR_MTTBAR-cut",
            f"YTTBAR{get_suffix()}": f"251021-jth-dataset-selection-with-ATLAS_TTBAR_YTTBAR{get_suffix()}",
            f"MTTBAR-PTT{get_suffix()}": f"251014-jth-dataset-selection-with-ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-PTT{get_suffix()}",
            #f"MTTBAR-YTTBAR{get_suffix()}": f"251014-jth-dataset-selection-with-ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR{get_suffix()}",
            f"MTTBAR-YTTBAR-cut": f"251105-jth-dataset-selection-with-ATLAS_TTBAR_MTTBAR-YTTBAR-cut",
        },
        "CMS": {
            f"YT{get_suffix()}": f"251022-jth-dataset-selection-with-CMS_TTBAR_YT{get_suffix()}",
            f"PTT{get_suffix()}": f"251022-jth-dataset-selection-with-CMS_TTBAR_PTT{get_suffix()}",
            #f"MTTBAR{get_suffix()}": f"251022-jth-dataset-selection-with-CMS_TTBAR_MTTBAR{get_suffix()}",
            f"MTTBAR-cut": f"251105-jth-dataset-selection-with-CMS_TTBAR_MTTBAR-cut",
            f"YTTBAR{get_suffix()}": f"251022-jth-dataset-selection-with-CMS_TTBAR_YTTBAR{get_suffix()}",
            #f"MTTBAR-YTTBAR{get_suffix()}": f"251022-jth-dataset-selection-with-CMS_TTBAR_MTTBAR-YTTBAR{get_suffix()}",
            f"MTTBAR-YTTBAR-cut": f"251105-jth-dataset-selection-with-CMS_TTBAR_MTTBAR-YTTBAR-cut",
        },
        "ATLAS_CMS": {
            f"YT{get_suffix()}": f"251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_YT{get_suffix()}",
            f"PTT{get_suffix()}": f"251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_PTT{get_suffix()}",
            #f"MTTBAR{get_suffix()}": f"251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_MTTBAR{get_suffix()}",
            f"MTTBAR-cut": f"251105-jth-dataset-selection-with-ATLAS_CMS_TTBAR_MTTBAR-cut",
            f"YTTBAR{get_suffix()}": f"251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_YTTBAR{get_suffix()}",
            #f"MTTBAR-YTTBAR{get_suffix()}": f"251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_MTTBAR-YTTBAR{get_suffix()}",
            f"MTTBAR-YTTBAR-cut": f"251105-jth-dataset-selection-with-ATLAS_CMS_TTBAR_MTTBAR-YTTBAR-cut",
            f"MTTBAR-PTT{get_suffix()}": f"251014-jth-dataset-selection-with-ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-PTT{get_suffix()}",
        }
    },
    "N3LO": {
        "ATLAS": {
            f"YT{get_suffix()}": f"251118-jth-dataset-selection-N3LO-with-ATLAS_TTBAR_YT{get_suffix()}",
            f"PTT{get_suffix()}": f"251118-jth-dataset-selection-N3LO-with-ATLAS_TTBAR_PTT{get_suffix()}",
            #f"MTTBAR{get_suffix()}": f"251118-jth-dataset-selection-N3LO-with-ATLAS_TTBAR_MTTBAR{get_suffix()}",
            f"MTTBAR-cut": f"251118-jth-dataset-selection-N3LO-with-ATLAS_TTBAR_MTTBAR-cut",
            f"YTTBAR{get_suffix()}": f"251118-jth-dataset-selection-N3LO-with-ATLAS_TTBAR_YTTBAR{get_suffix()}",
            f"MTTBAR-PTT{get_suffix()}": f"251118-jth-dataset-selection-N3LO-with-ATLAS_TTBAR_MTTBAR-PTT{get_suffix()}",
            #f"MTTBAR-YTTBAR{get_suffix()}": f"251118-jth-dataset-selection-N3LO-with-ATLAS_TTBAR_MTTBAR-YTTBAR{get_suffix()}",
            f"MTTBAR-YTTBAR-cut": f"251118-jth-dataset-selection-N3LO-with-ATLAS_TTBAR_MTTBAR-YTTBAR-cut",
        },
        "CMS": {
            f"YT{get_suffix()}": f"251118-jth-dataset-selection-N3LO-with-CMS_TTBAR_YT{get_suffix()}",
            f"PTT{get_suffix()}": f"251118-jth-dataset-selection-N3LO-with-CMS_TTBAR_PTT{get_suffix()}",
            #f"MTTBAR{get_suffix()}": f"251118-jth-dataset-selection-N3LO-with-CMS_TTBAR_MTTBAR{get_suffix()}",
            f"MTTBAR-cut": f"251118-jth-dataset-selection-N3LO-with-CMS_TTBAR_MTTBAR-cut",
            f"YTTBAR{get_suffix()}": f"251118-jth-dataset-selection-N3LO-with-CMS_TTBAR_YTTBAR{get_suffix()}",
            #f"MTTBAR-YTTBAR{get_suffix()}": f"251118-jth-dataset-selection-N3LO-with-CMS_TTBAR_MTTBAR-YTTBAR{get_suffix()}",
            f"MTTBAR-YTTBAR-cut": f"251118-jth-dataset-selection-N3LO-with-CMS_TTBAR_MTTBAR-YTTBAR-cut",
        },
        "ATLAS_CMS": {
            f"YT{get_suffix()}": f"251118-jth-dataset-selection-N3LO-with-ATLAS_CMS_TTBAR_YT{get_suffix()}",
            f"PTT{get_suffix()}": f"251118-jth-dataset-selection-N3LO-with-ATLAS_CMS_TTBAR_PTT{get_suffix()}",
            #f"MTTBAR{get_suffix()}": f"251118-jth-dataset-selection-N3LO-with-ATLAS_CMS_TTBAR_MTTBAR{get_suffix()}",
            f"MTTBAR-cut": f"251118-jth-dataset-selection-N3LO-with-ATLAS_CMS_TTBAR_MTTBAR-cut",
            f"YTTBAR{get_suffix()}": f"251118-jth-dataset-selection-N3LO-with-ATLAS_CMS_TTBAR_YTTBAR{get_suffix()}",
            #f"MTTBAR-YTTBAR{get_suffix()}": f"251118-jth-dataset-selection-N3LO-with-ATLAS_CMS_TTBAR_MTTBAR-YTTBAR{get_suffix()}",
            f"MTTBAR-YTTBAR-cut": f"251118-jth-dataset-selection-N3LO-with-ATLAS_CMS_TTBAR_MTTBAR-YTTBAR-cut",
            f"MTTBAR-PTT{get_suffix()}": f"251118-jth-dataset-selection-N3LO-with-ATLAS_TTBAR_MTTBAR-PTT{get_suffix()}",
        }
    }}




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

legend_elements = {}
latex_dict = {"YT": r"$y_t$", "PTT": r"$p_T^t$", "MTTBAR": r"$m_{t\bar{t}}$", "YTTBAR": r"$y_{t\bar{t}}$",
                "MTTBAR-PTT": r"$(m_{t\bar{t}}, p_T^t)$", "MTTBAR-YTTBAR": r"$(m_{t\bar{t}}, y_{t\bar{t}})$",
                "YT-NORM": r"$y_t$", "PTT-NORM": r"$p_T^t$", "MTTBAR-NORM": r"$m_{t\bar{t}}$", "YTTBAR-NORM": r"$y_{t\bar{t}}$",
                "MTTBAR-PTT-NORM": r"$(m_{t\bar{t}}, p_T^t)$", "MTTBAR-YTTBAR-NORM": r"$(m_{t\bar{t}}, y_{t\bar{t}})$",
              "MTTBAR-YT-NORM": r"$(m_{t\bar{t}}, y_t)$", "PTT-YT-NORM": r"$(p_T^t, y_t)$",
              "MTTBAR-cut": r"$m_{t\bar{t}}\;>343\,\mathrm{GeV}$",
              "MTTBAR-YTTBAR-cut": r"$(m_{t\bar{t}}\;>343\,\mathrm{GeV}, y_{t\bar{t}})$"}

def add_combined_analysis(ax, result_dir, combined_dir):

    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    for i, (obs, dir_name) in enumerate(combined_dir.items()):
        try:
            central_value = load_central_value(result_dir / dir_name / f"{dir_name}_central_value.dat")
            covmat = np.loadtxt(result_dir / dir_name / f"{dir_name}_covmat.dat")
        except FileNotFoundError:
            print(f"{dir_name} not found, skipping.")
            continue
        confidence_ellipse(ax, covmat, central_value, confidence_level=68, facecolor=colors_dict[obs], edgecolor=colors_dict[obs])

        if obs not in legend_elements:
            face_rgba = (*plt.cm.colors.to_rgba(colors[i])[:3], 0.3)
            edge_rgba = (*plt.cm.colors.to_rgba(colors[i])[:3], 1.0)
            legend_elements[obs] = patches.Patch(facecolor=face_rgba, edgecolor=edge_rgba, label=latex_dict[obs])


for order in combined_dirs.keys():
    combined_dir_order = combined_dirs[order]


    fig, axes = plt.subplots(3, 1,
                                 figsize=(9 , 5 * 2),
                                 squeeze=False)

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

        add_combined_analysis(axi, RESULT_DIR, combined_dir_order[experiment])

        axi.set_xlim(x_min, x_max)
        axi.set_ylim(y_min, y_max)

        axi.set_ylabel(r"$\alpha_s(m_Z)$")

        axi.text(0.03, 0.93, exp_label, transform=axi.transAxes, fontsize=17,
                 verticalalignment='top', bbox=props, color='black')
        axi.text(0.97, 0.05, rf"$\mathbf{{{order}}}+\mathbf{{MHOU}}$", transform=axi.transAxes, fontsize=17,
                 verticalalignment='bottom', bbox=props, color='black', horizontalalignment='right')

    legend_elements = list(legend_elements.values())
    fig.legend(handles=legend_elements, loc="upper center", fontsize=17, frameon=False, bbox_to_anchor=(0.5, 0.98),
              ncol=np.ceil(len(legend_elements) / 2))

    legend_elements = {}

    fig.savefig(f"251118_jth_observable_comparison_{order}_cut.pdf")






# get the unique legend elements




