import pathlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from matplotlib import patches, transforms, rc
from matplotlib.patches import Ellipse

# ---------------- Configuration ---------------- #
RESULT_DIR = pathlib.Path("./results")
USE_NORMALISED = False
ELLIPSE_MARGIN = 0.7
OUTPUT_FILE = "251001-jth-alphas_mtop_dataset_selection_abs.pdf"

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


def get_suffix():
    return "NORM" if USE_NORMALISED else ""


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
    ellipse.set_facecolor((*plt.cm.colors.to_rgba(facecolor if facecolor else "C0")[:3], 0.3))
    ellipse.set_edgecolor((*plt.cm.colors.to_rgba(kwargs.get("edgecolor", "black"))[:3], 1.0))

    ax.add_patch(ellipse)
    ax.scatter(mean[0], mean[1], marker="x", color="black")
    ax.grid(True)
    return width, height


def filter_results(result_dir):
    results = {}
    for dataset_dir in result_dir.iterdir():
        if "250927" not in dataset_dir.name or not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name.split("-with-")[1]
        chi2_df = pd.read_csv(dataset_dir / "chi2.txt", sep="\t", skip_blank_lines=True)
        chi2_df.columns = chi2_df.columns.str.strip()

        ndat = int(chi2_df["ndat"].values[0])
        chi2_ttbar = chi2_df["chi2 ttbar"].values[0]
        p_value_ttbar = 1 - scipy.stats.chi2.cdf(chi2_ttbar * ndat, ndat)

        if 0.05 < p_value_ttbar < 0.95:
            results[dataset_name] = {
                "p-value": p_value_ttbar,
                "chi2": chi2_df,
                "central_value": load_central_value(dataset_dir / f"{dataset_dir.name}_central_value.dat"),
                "covmat": np.loadtxt(dataset_dir / f"{dataset_dir.name}_covmat.dat"),
            }
        else:
            print(f"Skipping {dataset_name} with p={p_value_ttbar:.3f}")
    return results

def plot_single_experiment(ax, dataset_info, exp, obs):
    cv = dataset_info["central_value"]
    cov = dataset_info["covmat"]

    width, height = confidence_ellipse(
        ax, cov, cv, edgecolor="C0", facecolor="C0", confidence_level=68
    )
    # ax.set_xlim(cv[0] - ELLIPSE_MARGIN * width, cv[0] + ELLIPSE_MARGIN * width)
    # ax.set_ylim(cv[1] - ELLIPSE_MARGIN * height, cv[1] + ELLIPSE_MARGIN * height)



    # # add text box with chi2/ndat
    chi2_ttbar = dataset_info["chi2"]["chi2 ttbar"].values[0]
    chi2_tot = dataset_info["chi2"]["chi2 tot"].values[0]
    ndat = int(dataset_info["chi2"]["ndat"].values[0])
    textstr = r'$\chi^2_{t\bar{t}}=%.2f$, $n_{\mathrm{dat}}=%d$' % (chi2_ttbar, ndat)
    p_value = dataset_info["p-value"]
    textstr_pvalue = r'$p-\mathrm{val}=%.2f$' % (p_value)
    # these are matplotlib.patch.Patch properties
    props = dict(facecolor='none', edgecolor='none')
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, color='black')

    ax.text(0.05, 0.05, textstr_pvalue, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props, color='red' if p_value < 0.05 or p_value > 0.95 else 'black')

    return cv, width, height


def plot_statistical_average(ax_row, experiments, observables, results):
    avg_means, avg_widths, avg_heights = {}, {}, {}
    for j, obs in enumerate(observables):
        means, covs = [], []
        for exp in experiments:
            for dataset_name in results:
                if exp in dataset_name and dataset_name.endswith(obs):
                    if results[dataset_name]["p-value"] < 0.05 or results[dataset_name]["p-value"] > 0.95:
                        continue
                    means.append(results[dataset_name]["central_value"])
                    covs.append(results[dataset_name]["covmat"])
                    break

        if means:
            means, covs = np.array(means), np.array(covs)
            avg_mean = np.mean(means, axis=0)
            avg_cov = np.sum(covs, axis=0) / len(means)**2
            width, height = confidence_ellipse(ax_row[j], avg_cov, avg_mean, edgecolor="C1", facecolor="C1", confidence_level=68)
            avg_means[obs] = avg_mean
            avg_widths[obs] = width
            avg_heights[obs] = height
    return avg_means, avg_widths, avg_heights

def add_combined_analysis(ax_row, observables, result_dir):
    combined_dirs = {
        f"MTTBAR{get_suffix()}": f"251001-jth-dataset-selection-with-DIF_MTTBAR{get_suffix()}",
        f"PTT{get_suffix()}": f"251001-jth-dataset-selection-with-DIF_PTT{get_suffix()}",
        f"YT{get_suffix()}": f"251001-jth-dataset-selection-with-DIF_YT{get_suffix()}",
        f"YTTBAR{get_suffix()}": f"251001-jth-dataset-selection-with-DIF_YTTBAR{get_suffix()}",
    }

    for obs, dir_name in combined_dirs.items():
        if obs not in observables:
            continue
        j = observables.index(obs)
        central_value = load_central_value(result_dir / dir_name / f"{dir_name}_central_value.dat")
        covmat = np.loadtxt(result_dir / dir_name / f"{dir_name}_covmat.dat")
        confidence_ellipse(ax_row[j], covmat, central_value, edgecolor="C2", facecolor="C2", confidence_level=68)


# ---------------- Main ---------------- #
def main():
    results = filter_results(RESULT_DIR)

    experiments, observables = set(), set()
    for dataset_name in results.keys():
        if (USE_NORMALISED and "NORM" not in dataset_name) or (not USE_NORMALISED and "NORM" in dataset_name):
            continue
        if "DIF" in dataset_name:
            parts = dataset_name.split("_")
            observables.add(parts[-1])
            experiments.add("_".join(parts[:-2]))

    experiments, observables = sorted(experiments), sorted(observables)

    fig, axes = plt.subplots(len(experiments) + 1, len(observables),
                             figsize=(3 * len(observables), 3 * (len(experiments) + 1)),
                             squeeze=False)

    # placeholders for limits, to be updated dynamically later
    xlims_left = 1e4 * np.ones((len(experiments) + 1, len(observables)), dtype=float)
    xlims_right = np.zeros((len(experiments) + 1, len(observables)), dtype=float)
    ylims_left = 1e4 * np.ones((len(experiments) + 1, len(observables)), dtype=float)
    ylims_right = np.zeros((len(experiments) + 1, len(observables)), dtype=float)

    for i, exp in enumerate(experiments):
        for j, obs in enumerate(observables):
            ax = axes[i, j]
            dataset_name = f"{exp}_DIF_{obs}"

            if dataset_name in results:

                cv, width, height = plot_single_experiment(ax, results[dataset_name], exp, obs)
                xlims_left[i, j] = cv[0] - ELLIPSE_MARGIN * width
                xlims_right[i, j] = cv[0] + ELLIPSE_MARGIN * width
                ylims_left[i, j] = cv[1] - ELLIPSE_MARGIN * height
                ylims_right[i, j] = cv[1] + ELLIPSE_MARGIN * height
            else:
                ax.set_visible(False)

    # add statistical average to last row
    avg_means, avg_widths, avg_heights = plot_statistical_average(axes[-1], experiments, observables, results)
    for j, obs in enumerate(observables):
        if obs in avg_means:
            cv, width, height = avg_means[obs], avg_widths[obs], avg_heights[obs]
            xlims_left[-1, j] = cv[0] - width
            xlims_right[-1, j] = cv[0] + width
            ylims_left[-1, j] = cv[1] - height
            ylims_right[-1, j] = cv[1] + height

    # add combined analysis to the last row
    add_combined_analysis(axes[-1], observables, RESULT_DIR)

    # --- compute grid edges ---
    left_edge = min(ax.get_position().x0 for row in axes for ax in row if ax.get_visible())
    top_edge = max(ax.get_position().y1 for row in axes for ax in row if ax.get_visible())

    # --- add row labels (y-axis) ---
    experiments += ["Combination"]
    for i, exp in enumerate(experiments):
        y_center = (axes[i, 0].get_position().y0 + axes[i, 0].get_position().y1) / 2
        fig.text(
            left_edge - 0.07, y_center + 0.02, y_label_dict[exp], va="center", ha="right", rotation=90, fontsize=14
        )

    # --- add column labels (x-axis, at the top) ---

    for j, obs in enumerate(observables):
        x_center = (axes[0, j].get_position().x0 + axes[0, j].get_position().x1) / 2
        fig.text(
            x_center + 0.035, top_edge + 0.02, x_label_dict[obs], va="bottom", ha="center", fontsize=14
        )


    for i in range(len(experiments)):
        for j in range(len(observables)):

            ax = axes[i, j]
            if not ax.get_visible():
                continue

            ax.set_xlabel(r"$m_t\;\mathrm{[GeV]}$")
            ax.set_ylabel(r"$\alpha_s(m_Z)$")

            keep_y = j == 0 or axes[
                i, j - 1].get_visible() == False  # keep y labels for first row or if left subplot is not visible
            keep_x = i == len(experiments) - 1 or axes[i + 1, j].get_visible() == False

            if not keep_y:
                ax.set_ylabel("")
                ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            # Hide x labels/ticks except for bottom row
            if not keep_x:
                ax.set_xlabel("")
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # set same x and y limits for all plots

    # set same x and y limits across columns and rows
    for i in range(len(experiments)):
        for j in range(len(observables)):
            if axes[i, j].get_visible():
                axes[i, j].set_xlim(np.min(xlims_left[:, j]), np.max(xlims_right[:, j]))
                axes[i, j].set_ylim(np.min(ylims_left[i, :]), np.max(ylims_right[i, :]))

    # Move subplots closer together


    plt.tight_layout(rect=[0.1, 0.1, 0.95, 0.9])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    legend_elements = [
        patches.Patch(facecolor="C0", edgecolor="C0", alpha=0.3, label=r"$\mathrm{Individual\;analysis}$"),
        patches.Patch(facecolor="C1", edgecolor="C1", alpha=0.3, label=r"$\mathrm{Statistical\;average}$"),
        patches.Patch(facecolor="C2", edgecolor="C2", alpha=0.3, label=r"$\mathrm{Combined\;analysis}$"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, 0.06),
               ncol=len(legend_elements), fontsize=17, frameon=False)

    plt.savefig(OUTPUT_FILE)


if __name__ == "__main__":
    main()

