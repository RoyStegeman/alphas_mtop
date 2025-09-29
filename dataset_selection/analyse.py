import pathlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd

from matplotlib import patches, transforms, rc
# use latex
# rc('text', usetex=True)
# rc('font', family='serif', size=12)
from matplotlib.patches import Ellipse

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

    ax.set_xlim(mean[0] - width, mean[0] + width)
    ax.set_ylim(mean[1] - height, mean[1] + height)



result_dir = pathlib.Path("./results")
results = {}
for dataset_dir in result_dir.iterdir():
    if "250926" in dataset_dir.name:
        continue
    if dataset_dir.is_dir():
        print(f"Analyzing results in {dataset_dir}")
        # split the name of the dataset_dir
        dataset_name = dataset_dir.name.split("-with-")[1]
        results[dataset_name] = {}
        with open(dataset_dir / f"{dataset_dir.name}_central_value.dat") as f:
            line = f.readline()#.split(':')[1].strip()
            central_value = np.fromstring(line.strip().strip('[]'), sep=' ')
            results[dataset_name]["central_value"] = central_value

        covmat = np.loadtxt(dataset_dir / f"{dataset_dir.name}_covmat.dat")
        results[dataset_name]["covmat"] = covmat

        # read chi2 info
        chi2_df = pd.read_csv(dataset_dir / 'chi2.txt', sep='\t', skip_blank_lines=True)
        chi2_df.columns = chi2_df.columns.str.strip()
        results[dataset_name]["chi2"] = chi2_df
        


# collect all experiments and observables
experiments = set()
observables = set()
for dataset_name in results.keys():
    # if "NORM" not in dataset_name:
    #     continue
    if "DIF" in dataset_name:
        parts = dataset_name.split("_")
        observables.add(parts[-1])
        experiment_name = "_".join(parts[:-2])
        experiments.add(experiment_name)
    # elif "TOT" in dataset_name:
    #     parts = dataset_name.split("_TOT_X-SEC")
    #     observables.add("TOT_X-SEC")
    #     experiment_name = parts[0]
    #     experiments.add(experiment_name)

# produce a grid of plots with experiments as columns and observables as rows

experiments = sorted(experiments)
observables = sorted(observables)


fig, axes = plt.subplots(
    len(observables), len(experiments) + 1,
    figsize=(4 * (len(experiments) + 1), 4 * len(observables)),
    squeeze=False
)

for i, obs in enumerate(observables):
    for j, exp in enumerate(experiments):
        ax = axes[i, j]
        ax.set_xlabel(r"$m_t$", fontsize=14)
        ax.set_ylabel(r"$\alpha_s$", fontsize=14)

        # Find the matching dataset_name
        for dataset_name in results:
            if exp in dataset_name and obs in dataset_name:
                central_value = results[dataset_name]["central_value"]
                covmat = results[dataset_name]["covmat"]
                
                chi2_df = results[dataset_name]["chi2"]
                ndat = int(chi2_df["ndat"].values[0])
                chi2_ttbar = chi2_df["chi2 ttbar"].values[0]
                chi2_tot = chi2_df["chi2 tot"].values[0]

                confidence_ellipse(
                    ax, covmat, central_value,
                    edgecolor="C0", facecolor="C0", confidence_level=68
                )

                # # add text box with chi2/ndat
                textstr = r'$\chi^2_{\mathrm{ttbar}}=%.2f$, $\chi^2_{\mathrm{tot}}=%.2f$' % (chi2_ttbar, chi2_tot)
                textstr_ndat = r'$N_{\mathrm{dat}}=%d$' % (ndat, )
                # these are matplotlib.patch.Patch properties
                props = dict(facecolor='none', edgecolor='none')
                # place a text box in upper left in axes coords
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props, color='red' if chi2_ttbar > 1.5 else 'black')

                ax.text(0.05, 0.05, textstr_ndat, transform=ax.transAxes, fontsize=10,
                        verticalalignment='bottom', bbox=props)

                break
        else:
            ax.set_visible(False)  # Hide unused subplots

# plot the average of of the ellipses at the end of each row
for i, obs in enumerate(observables):
    ax = axes[i, -1]  # last column
    means = []
    covs = []
    for j, exp in enumerate(experiments):
        for dataset_name in results:
            if exp in dataset_name and obs in dataset_name:
                if results[dataset_name]["chi2"]["chi2 ttbar"].values[0] > 1.5:
                    continue
                means.append(results[dataset_name]["central_value"])
                covs.append(results[dataset_name]["covmat"])
                break
    if means:
        means = np.array(means)
        covs = np.array(covs)
        avg_mean = np.mean(means, axis=0)
        avg_cov = np.mean(covs, axis=0)
        confidence_ellipse(
            ax, avg_cov, avg_mean,
            edgecolor="C1", facecolor="C1", confidence_level=68
        )
        ax.set_xlabel(r"$m_t$", fontsize=14)
        ax.set_ylabel(r"$\alpha_s$", fontsize=14)
        ax.set_visible(True)
    else:
        ax.set_visible(False)

plt.tight_layout(rect=[0.1, 0.1, 0.95, 0.9])  # leave margins for labels

label_dict = { "MTTBAR": r"$d\sigma/d m_{t\bar{t}}$", "MTTBAR-NORM": r"$1/\sigma d\sigma m_{t\bar{t}}$",
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


# --- compute grid edges ---
left_edge = min(ax.get_position().x0 for row in axes for ax in row if ax.get_visible())
top_edge  = max(ax.get_position().y1 for row in axes for ax in row if ax.get_visible())

# --- add row labels (y-axis) ---
for i, obs in enumerate(observables):
    y_center = (axes[i, 0].get_position().y0 + axes[i, 0].get_position().y1) / 2
    fig.text(
        left_edge - 0.04, y_center, label_dict[obs], va="center", ha="right", rotation=90
    )

# --- add column labels (x-axis, at the top) ---
experiments += ["Average"]
for j, exp in enumerate(experiments):
    x_center = (axes[0, j].get_position().x0 + axes[0, j].get_position().x1) / 2
    fig.text(
        x_center, top_edge + 0.02, exp, va="bottom", ha="center"
    )

plt.savefig("250929-jth-alphas_mtop_dataset_selection.pdf")