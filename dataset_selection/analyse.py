import pathlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd

from matplotlib import patches, transforms, rc
from matplotlib.patches import Ellipse

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 12})
rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})

x_label_dict = { "MTTBAR": r"$d\sigma/d m_{t\bar{t}}$", "MTTBAR-NORM": r"$1/\sigma d\sigma m_{t\bar{t}}$",
               "MTTBAR-PTT": r"$d^2\sigma/dm_{t\bar{t}}dp_T^t$",
               "MTTBAR-PTT-NORM": r"$1/\sigma d^2\sigma/dm_{t\bar{t}}dp_T^t$",
               "MTTBAR-YT-NORM": r"$1/\sigma d^2\sigma/dm_{t\bar{t}}dy_{t\bar{t}}$",
               "MTTBAR-YTTBAR": r"$d^2\sigma/dm_{t\bar{t}}dy_{t\bar{t}}$",
               "MTTBAR-YTTBAR-NORM": r"$1/\sigma d^2\sigma/dm_{t\bar{t}}dy_{t\bar{t}}$",
                "PTT": r"$d\sigma/dp_T^t$",
               "PTT-NORM": r"$1/\sigma d\sigma/d p_T^t$",
               "PTT-YT-NORM": r"$1/\sigma d^2\sigma/dp_T^t dy_t$",
               "YT": r"$d\sigma/d y_t$",
               "YT-NORM": r"$1/\sigma d\sigma/d y_t$",
               "YTTBAR": r"$d\sigma/dy_{t\bar{t}}$",
               "YTTBAR-NORM": r"$1/\sigma d\sigma/d y_{t\bar{t}}$"}

y_label_dict = {"ATLAS_TTBAR_13TEV_HADR": r"$\mathrm{ATLAS}\;t\bar{t}\;13\;\mathrm{TeV}\;\mathrm{hadr.}$",
"ATLAS_TTBAR_13TEV_LJ": r"$\mathrm{ATLAS}\;t\bar{t}\;13\;\mathrm{TeV}\;\ell+j$",
"ATLAS_TTBAR_8TEV_2L": r"$\mathrm{ATLAS}\;t\bar{t}\;8\;\mathrm{TeV}\;2\ell$",
"ATLAS_TTBAR_8TEV_LJ": r"$\mathrm{ATLAS}\;t\bar{t}\;8\;\mathrm{TeV}\;\ell + j$",
"CMS_TTBAR_13TEV_2L": r"$\mathrm{CMS}\;t\bar{t}\;13\;\mathrm{TeV}\;2\ell$",
"CMS_TTBAR_13TEV_LJ": r"$\mathrm{CMS}\;t\bar{t}\;13\;\mathrm{TeV}\;\ell + j$",
"CMS_TTBAR_8TEV_2L": r"$\mathrm{CMS}\;t\bar{t}\;8\;\mathrm{TeV}\;2\ell$",
"CMS_TTBAR_8TEV_LJ": r"$\mathrm{CMS}\;t\bar{t}\;8\;\mathrm{TeV}\;\ell + j$",
"Combination": r"$\mathrm{Combination}$"}

use_normalised = True

def get_suffix():
    return "NORM" if use_normalised else ""

def confidence_ellipse(ax, cov, mean, facecolor=None, confidence_level=95, **kwargs):
    """
    Draws a confidence ellipse for a 2D Gaussian defined by mean and cov.
    """
    # eigen-decomposition
    eig_val, eig_vec = np.linalg.eigh(cov)
    order = np.argsort(eig_val)[::-1]
    eig_val, eig_vec = eig_val[order], eig_vec[:, order]

    # chi2 quantile for desired CL
    chi2_qnt = scipy.stats.chi2.ppf(confidence_level / 100.0, 2)

    # ellipse radii = sqrt(chi2 * eigenvalues)
    width, height = 2 * np.sqrt(chi2_qnt * eig_val)

    # angle of ellipse (in degrees)
    angle = np.degrees(np.arctan2(*eig_vec[:, 0][::-1]))

    # single ellipse with transparent face and opaque edge
    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        facecolor=facecolor,
        edgecolor=kwargs.get("edgecolor", "black"),
        alpha=0.3,  # transparency applies only to facecolor
        linewidth=2,
    )

    # make sure edge is opaque
    ellipse.set_edgecolor(kwargs.get("edgecolor", "black"))
    ellipse.set_facecolor(facecolor if facecolor is not None else "none")
    ellipse.set_alpha(None)  # remove global alpha
    ellipse.set_facecolor((*plt.cm.colors.to_rgba(facecolor if facecolor else "C0")[:3], 0.3))
    ellipse.set_edgecolor((*plt.cm.colors.to_rgba(kwargs.get("edgecolor", "black"))[:3], 1.0))


    ax.add_patch(ellipse)

    ax.scatter(mean[0], mean[1], marker='x', color='black')
    # set grid
    ax.grid(True)

    # expand limits to make sure ellipse is visible

    # ax.set_xlim(mean[0] - 0.7 * width, mean[0] + 0.7*  width)
    # ax.set_ylim(mean[1] - 0.7* height, mean[1] + 0.7* height)

    return width, height


result_dir = pathlib.Path("./results")
results = {}
for dataset_dir in result_dir.iterdir():
    if "250927" not in dataset_dir.name:
        continue
    if dataset_dir.is_dir():
        print(f"Analyzing results in {dataset_dir}")
        # split the name of the dataset_dir
        dataset_name = dataset_dir.name.split("-with-")[1]

        # read chi2 info
        chi2_df = pd.read_csv(dataset_dir / 'chi2.txt', sep='\t', skip_blank_lines=True)
        chi2_df.columns = chi2_df.columns.str.strip()


        # compute p value for chi2_ttbar
        ndat = int(chi2_df["ndat"].values[0])
        chi2_ttbar = chi2_df["chi2 ttbar"].values[0]

        p_value_ttbar = 1 - scipy.stats.chi2.cdf(chi2_ttbar * ndat, ndat)

        # continue if p_value_ttbar < 0.05 or p_value_ttbar > 0.95
        if p_value_ttbar < 0.05 or p_value_ttbar > 0.95:
            print(f"Skipping dataset {dataset_name} with p-value {p_value_ttbar:.3f}")
            continue

        results[dataset_name] = {}
        results[dataset_name]["p-value"] = p_value_ttbar
        results[dataset_name]["chi2"] = chi2_df


        with open(dataset_dir / f"{dataset_dir.name}_central_value.dat") as f:
            line = f.readline()#.split(':')[1].strip()
            central_value = np.fromstring(line.strip().strip('[]'), sep=' ')
            results[dataset_name]["central_value"] = central_value

        covmat = np.loadtxt(dataset_dir / f"{dataset_dir.name}_covmat.dat")
        results[dataset_name]["covmat"] = covmat



# collect all experiments and observables
experiments = set()
observables = set()
for dataset_name in results.keys():
    if use_normalised and "NORM" not in dataset_name:
        continue
    if not use_normalised and "NORM" in dataset_name:
        continue
    if "DIF" in dataset_name:
        parts = dataset_name.split("_")
        observables.add(parts[-1])
        experiment_name = "_".join(parts[:-2])
        experiments.add(experiment_name)

experiments = sorted(experiments)
observables = sorted(observables)


fig, axes = plt.subplots(
    len(experiments) + 1, len(observables),
    figsize=(3 * len(observables), 3 * (len(experiments) + 1)),
    squeeze=False
)

# placeholders for limits, to be updated dynamically later
xlims_left = 1e4 * np.ones((len(experiments) + 1, len(observables)), dtype=float)
xlims_right = np.zeros((len(experiments) + 1, len(observables)), dtype=float)
ylims_left = 1e4 * np.ones((len(experiments) + 1, len(observables)), dtype=float)
ylims_right = np.zeros((len(experiments) + 1, len(observables)), dtype=float)

for i, exp in enumerate(experiments):
    for j, obs in enumerate(observables):
        ax = axes[i, j]
        ax.set_xlabel(r"$m_t$", fontsize=14)
        ax.set_ylabel(r"$\alpha_s$", fontsize=14)

        # Find the matching dataset_name
        for dataset_name in results:
            dataset_name_obs = dataset_name.split("_")[-1]
            if exp in dataset_name and obs == dataset_name_obs:
                central_value = results[dataset_name]["central_value"]
                covmat = results[dataset_name]["covmat"]
                
                chi2_df = results[dataset_name]["chi2"]
                ndat = int(chi2_df["ndat"].values[0])
                chi2_ttbar = chi2_df["chi2 ttbar"].values[0]
                chi2_tot = chi2_df["chi2 tot"].values[0]
                p_value = results[dataset_name]["p-value"]

                width, height = confidence_ellipse(
                    ax, covmat, central_value,
                    edgecolor="C0", facecolor="C0", confidence_level=68
                )

                # store limits
                xlims_left[i, j] = central_value[0] - 0.7 * width
                xlims_right[i, j] = central_value[0] + 0.7 * width
                ylims_left[i, j] = central_value[1] - 0.7 * height
                ylims_right[i, j] = central_value[1] + 0.7 * height

                # # add text box with chi2/ndat
                textstr = r'$\chi^2_{t\bar{t}}=%.2f$, $\chi^2_{\mathrm{tot}}=%.2f$' % (chi2_ttbar, chi2_tot)
                textstr += '\n'
                textstr += r'$n_{\mathrm{dat}}=%d$' % (ndat)
                textstr_pvalue = r'$p-\mathrm{val}=%.2f$' % (p_value)
                # these are matplotlib.patch.Patch properties
                props = dict(facecolor='none', edgecolor='none')
                # place a text box in upper left in axes coords
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props, color='black')

                ax.text(0.05, 0.05, textstr_pvalue, transform=ax.transAxes, fontsize=10,
                        verticalalignment='bottom', bbox=props, color='red' if p_value < 0.05 or p_value > 0.95 else 'black')

                break
        else:
            ax.set_visible(False)  # Hide unused subplots


# plot the average of of the ellipses at the end of each row
for i, obs in enumerate(observables):
    ax = axes[-1, i]  # last row
    means = []
    covs = []
    for j, exp in enumerate(experiments):
        for dataset_name in results:
            dataset_name_obs = dataset_name.split("_")[-1]
            if exp in dataset_name and obs == dataset_name_obs:
                if results[dataset_name]["p-value"] < 0.05 or results[dataset_name]["p-value"] > 0.95:
                    continue
                means.append(results[dataset_name]["central_value"])
                covs.append(results[dataset_name]["covmat"])
                break
    if means:
        n_exp = len(experiments)
        means = np.array(means)
        covs = np.array(covs)
        avg_mean = np.mean(means, axis=0)

        avg_cov = np.sum(covs, axis=0) / len(means)**2
        width, height = confidence_ellipse(
            ax, avg_cov, avg_mean,
            edgecolor="C1", facecolor="C1", confidence_level=68
        )
        ax.set_xlabel(r"$m_t$", fontsize=14)
        ax.set_ylabel(r"$\alpha_s$", fontsize=14)
        ax.set_visible(True)

        # store limits
        xlims_left[-1, i] = avg_mean[0] - width
        xlims_right[-1, i] = avg_mean[0] + width
        ylims_left[-1, i] = avg_mean[1] - height
        ylims_right[-1, i] = avg_mean[1] + height

    else:
        ax.set_visible(False)

# set same x and y limits for all plots

for i in range(len(experiments) + 1):
    for j in range(len(observables)):
        if axes[i, j].get_visible():
            axes[i, j].set_xlim(np.min(xlims_left[:, j]), np.max(xlims_right[:, j]))
            axes[i, j].set_ylim(np.min(ylims_left[i, :]), np.max(ylims_right[i, :]))

plt.tight_layout(rect=[0.1, 0.1, 0.95, 0.9])  # leave margins for labels

# --- compute grid edges ---
left_edge = min(ax.get_position().x0 for row in axes for ax in row if ax.get_visible())
top_edge  = max(ax.get_position().y1 for row in axes for ax in row if ax.get_visible())

# --- add row labels (y-axis) ---
experiments += ["Combination"]
for i, exp in enumerate(experiments):
    y_center = (axes[i, 0].get_position().y0 + axes[i, 0].get_position().y1) / 2
    fig.text(
        left_edge - 0.07, y_center, y_label_dict[exp], va="center", ha="right", rotation=90, fontsize=14
    )

# --- add column labels (x-axis, at the top) ---

for j, obs in enumerate(observables):
    x_center = (axes[0, j].get_position().x0 + axes[0, j].get_position().x1) / 2
    fig.text(
        x_center, top_edge + 0.02, x_label_dict[obs], va="bottom", ha="center", fontsize=14
    )

for i in range(len(experiments)):
    for j in range(len(observables)):

        ax = axes[i, j]
        if not ax.get_visible():
            continue

        keep_y = j == 0  or axes[i, j-1].get_visible() == False  # keep y labels for first row or if left subplot is not visible
        keep_x = i == len(experiments) - 1 or axes[i+1, j].get_visible() == False
        # Hide y labels/ticks except for leftmost column
        if not keep_y:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        # Hide x labels/ticks except for bottom row
        if not keep_x:
            ax.set_xlabel("")
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# Move subplots closer together
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# add combined tcm analysis in the bottom left plot


combined_dirs = {
    f"MTTBAR-{get_suffix()}": f"250930-jth-dataset-selection-with-DIF_MTTBAR-{get_suffix()}",
    f"PTT-{get_suffix()}": f"250930-jth-dataset-selection-with-DIF_PTT-{get_suffix()}",
    f"YT-{get_suffix()}": f"250930-jth-dataset-selection-with-DIF_YT-{get_suffix()}",
    f"YTTBAR-{get_suffix()}": f"250930-jth-dataset-selection-with-DIF_YTTBAR-{get_suffix()}",
}

for obs, dir_name in combined_dirs.items():

    if obs not in observables:
        continue  # skip if this observable isn't in the current grid

    j = observables.index(obs)  # column index from the name of the distribution

    combined_dir = result_dir / dir_name
    with open(combined_dir / f"{dir_name}_central_value.dat") as f:
        line = f.readline()
        central_value = np.fromstring(line.strip().strip('[]'), sep=' ')
    covmat = np.loadtxt(combined_dir / f"{dir_name}_covmat.dat")

    confidence_ellipse(
        axes[-1, j], covmat, central_value,
        edgecolor="C2", facecolor="C2", confidence_level=68
    )

# add a legend at the bottom of the figure
legend_elements = [
    patches.Patch(facecolor='C0', edgecolor='C0', alpha=0.3, label=r'$\mathrm{Single\;experimental\;dataset}$'),
    patches.Patch(facecolor='C1', edgecolor='C1', alpha=0.3, label=r'$\mathrm{Statistical\;average}$'),
    patches.Patch(facecolor='C2', edgecolor='C2', alpha=0.3, label=r'$\mathrm{Combined\;TCM\;analysis}$'),
]
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.06), ncol=len(legend_elements), fontsize=17, frameon=False)
# save figure


plt.savefig("250930-jth-alphas_mtop_dataset_selection_norm_only_filtered_v2.pdf")