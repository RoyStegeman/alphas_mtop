from validphys.api import API
from validphys.core import CommonDataSpec
import numpy as np
import nnpdf_data
import yaml
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 16})
rc('text', usetex=True)


baseline_fit = "250927-jth-dataset-selection-iter2"

# ATLAS
atlas_fits = {
    "YT": "251021-jth-dataset-selection-with-ATLAS_TTBAR_YT",
    "PTT": "251021-jth-dataset-selection-with-ATLAS_TTBAR_PTT",
    "MTTBAR": "251021-jth-dataset-selection-with-ATLAS_TTBAR_MTTBAR",
    "MTTBAR-cut": "251105-jth-dataset-selection-with-ATLAS_TTBAR_MTTBAR-cut",
    "YTTBAR": "251021-jth-dataset-selection-with-ATLAS_TTBAR_YTTBAR",
    "MTTBAR-PTT": "251014-jth-dataset-selection-with-ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-PTT",
    "MTTBAR-YTTBAR": "251014-jth-dataset-selection-with-ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR",
    "MTTBAR-YTTBAR-cut": "251105-jth-dataset-selection-with-ATLAS_TTBAR_MTTBAR-YTTBAR-cut",
}

# CMS
cms_fits = {
    "YT": "251022-jth-dataset-selection-with-CMS_TTBAR_YT",
    "PTT": "251022-jth-dataset-selection-with-CMS_TTBAR_PTT",
    "MTTBAR": "251022-jth-dataset-selection-with-CMS_TTBAR_MTTBAR",
    "MTTBAR-cut": "251105-jth-dataset-selection-with-CMS_TTBAR_MTTBAR-cut",
    "YTTBAR": "251022-jth-dataset-selection-with-CMS_TTBAR_YTTBAR",
    "MTTBAR-YTTBAR": "251022-jth-dataset-selection-with-CMS_TTBAR_MTTBAR-YTTBAR",
    "MTTBAR-YTTBAR-cut": "251105-jth-dataset-selection-with-CMS_TTBAR_MTTBAR-YTTBAR-cut",
}

# ATLAS + CMS
atlas_cms_fits = {
    "YT": "251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_YT",
    "PTT": "251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_PTT",
    "MTTBAR": "251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_MTTBAR",
    "MTTBAR-cut": "251105-jth-dataset-selection-with-ATLAS_CMS_TTBAR_MTTBAR-cut",
    "YTTBAR": "251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_YTTBAR",
    "MTTBAR-YTTBAR": "251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_MTTBAR-YTTBAR",
    "MTTBAR-YTTBAR-cut": "251105-jth-dataset-selection-with-ATLAS_CMS_TTBAR_MTTBAR-YTTBAR-cut"
}

label_dict = {"ATLAS_TTBAR_13TEV_HADR": r"$\mathrm{ATLAS}\;t\bar{t}\;13\;\mathrm{TeV}\;\mathrm{hadr.}$",
"ATLAS_TTBAR_13TEV_LJ": r"$\mathrm{ATLAS}\;t\bar{t}\;13\;\mathrm{TeV}\;\ell+j$",
"ATLAS_TTBAR_8TEV_2L": r"$\mathrm{ATLAS}\;t\bar{t}\;8\;\mathrm{TeV}\;2\ell$",
"ATLAS_TTBAR_8TEV_LJ": r"$\mathrm{ATLAS}\;t\bar{t}\;8\;\mathrm{TeV}\;\ell + j$",
"CMS_TTBAR_13TEV_2L": r"$\mathrm{CMS}\;t\bar{t}\;13\;\mathrm{TeV}\;2\ell$",
"CMS_TTBAR_13TEV_LJ": r"$\mathrm{CMS}\;t\bar{t}\;13\;\mathrm{TeV}\;\ell + j$",
"CMS_TTBAR_8TEV_2L": r"$\mathrm{CMS}\;t\bar{t}\;8\;\mathrm{TeV}\;2\ell$",
"CMS_TTBAR_8TEV_LJ": r"$\mathrm{CMS}\;t\bar{t}\;8\;\mathrm{TeV}\;\ell + j$",
"CMS_TTBAR_13TEV_2L_138FB-1": r"$\mathrm{CMS}\;t\bar{t}\;13\;\mathrm{TeV}\;2\ell\;138\;\mathrm{fb}^{-1}$",
"Combination": r"$\mathrm{Combination}$"}

x_label_dict = {"y_ttBar": r"$|y_{t\bar{t}}|$", "pT_t": r"$p_{T,t}\,\rm{[GeV]}$", "m_ttBar": r"$m_{t\bar{t}}\,\rm{[GeV]}$",
                "y_t": r"$|y_{t}|$"}

def unwrap_bins(x):
    """
    Make bin edges monotonic by shifting wrapped segments upward.
    Works for any number of drops.
    """
    x = np.array(x, dtype=float)
    unwrapped = x.copy()
    shift = 0.0

    for i in range(1, len(x)):
        if x[i] < x[i-1]:  # a drop → new wrapped segment
            shift += x[i-1]  # add previous max
        unwrapped[i] += shift

    return unwrapped

def split_on_wraps(x_raw, y):
    """
    Split x,y into continuous segments wherever x_raw decreases.
    Returns a list of (x_segment, y_segment).
    """
    segments = []
    start = 0

    for i in range(1, len(x_raw)):
        if x_raw[i] < x_raw[i-1]:     # found a wrap
            segments.append((x_raw[start:i], y[start:i]))
            start = i

    # last segment
    segments.append((x_raw[start:], y[start:]))
    return segments

def split_indices_on_wraps(x_raw):
    """Return list of (start, end) indices for each continuous block."""
    idx = [0]
    for i in range(1, len(x_raw)):
        if x_raw[i] < x_raw[i-1]:
            idx.append(i)
    idx.append(len(x_raw))
    return [(idx[i], idx[i+1]) for i in range(len(idx)-1)]

def get_wrap_indices(x_raw):
    """Return a list of indices i where x_raw[i] < x_raw[i-1]."""
    return [i for i in range(1, len(x_raw)) if x_raw[i] < x_raw[i-1]]

for fits in [atlas_cms_fits]:

    for obs, fit_name in fits.items():

        if fits == atlas_fits:
            suffix = "ATLAS"
        elif fits == cms_fits:
            suffix = "CMS"
        else:
            suffix = "ATLAS_CMS"



        dict_fit = dict(theoryid=40010006, use_cuts="internal", with_shift=False)

        fit = API.fit(fit=fit_name)
        dataset_inputs = fit.as_input()["dataset_inputs"]

        # metadata
        #
        # metadata = parse_new_metadata(metadata_path, observable_name, variant=variant)
        # return CommonDataSpec(setname, metadata)

        # filter only ttbar datasets
        dataset_inputs_ttbar = [ds for ds in dataset_inputs if "TTBAR" in ds["dataset"]]
        for ds in dataset_inputs_ttbar:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1], })
            ax_main = axes[0]
            ax_ratio = axes[1]

            # load the metadata
            try:
                metadata = nnpdf_data.load_dataset_metadata(ds["dataset"], variant=None)
            except FileNotFoundError:
                import pdb; pdb.set_trace()
            data, theory = API.results(dataset_input=ds, pdf=fit_name, pdfs=[fit_name], **dict_fit)
            _, theory_prefit = API.results(dataset_input=ds, pdf=baseline_fit, pdfs=[baseline_fit], **dict_fit)

            observable = metadata.plotting.plot_x
            with open(metadata.path_kinematics) as f:
                kinematics = yaml.safe_load(f)
                bin_mins = [bin[observable]["min"] for bin in kinematics["bins"]]
                bin_max = [bin[observable]["max"] for bin in kinematics["bins"]]


            x_label = metadata.plotting.x_label
            y_label = metadata.plotting.y_label
            #title = metadata.plotting.dataset_label
            linear = metadata.plotting.x_scale.name == "linear"

            #ax_main.set_title(title)
            ax_main.set_title(label_dict[ds["dataset"].split("_DIF")[0]])
            ax_main.set_ylabel(y_label)
            ax_main.set_xticklabels([])
            #ax_main.set_xscale('log' if not linear else 'linear')






            n_data = len(data)
            # x = np.array(bin_mins + [bin_max[-1]])
            # Original and unwrapped x
            x_raw = np.array(bin_mins + [bin_max[-1]])
            x = unwrap_bins(x_raw)



            data_y = np.concatenate([data.central_value, [data.central_value[-1]]])
            theory_y = np.concatenate([theory.central_value, [theory.central_value[-1]]])
            theory_prefit_y = np.concatenate([theory_prefit.central_value, [theory_prefit.central_value[-1]]])

            for i in range(len(data.central_value)):
                xi = x[i]
                width = x[i + 1] - xi
                cv = data.central_value[i]
                se = data.std_error[i]
                rect = Rectangle((xi, cv - se), width, 2.0 * se, facecolor="C0", alpha=0.3, edgecolor=None)
                ax_main.add_patch(rect)
            for i in range(len(theory.central_value)):
                xi = x[i]
                width = x[i + 1] - xi
                cv = theory.central_value[i]
                se = theory.std_error[i]
                rect = Rectangle((xi, cv - se), width, 2.0 * se, facecolor="C1", alpha=0.3, edgecolor=None)
                ax_main.add_patch(rect)
            for i in range(len(theory_prefit.central_value)):
                xi = x[i]
                width = x[i + 1] - xi
                cv = theory_prefit.central_value[i]
                se = theory_prefit.std_error[i]
                rect = Rectangle((xi, cv - se), width, 2.0 * se, facecolor="C2", alpha=0.3, edgecolor=None)
                ax_main.add_patch(rect)



            # Identify continuous segments
            segments = split_indices_on_wraps(x_raw)

            wraps = get_wrap_indices(x_raw)

            for i in wraps:
                xline = x[i]  # unwrapped coordinate where the jump happened
                ax_main.axvline(xline, color='k', linestyle='--', alpha=0.5)
                ax_ratio.axvline(xline, color='k', linestyle='--', alpha=0.5)

            # Plot step functions WITHOUT reconnecting the segments
            for start, end in segments:
                ax_main.step(x[start:end], data_y[start:end], where='post', color="C0")
                ax_main.step(x[start:end], theory_y[start:end], where='post', color="C1")
                ax_main.step(x[start:end], theory_prefit_y[start:end], where='post', color="C2")


            legend_handles = [
                Patch(facecolor="C0", alpha=0.3, edgecolor="C0", label=r"${\rm Data}$"),
                Patch(facecolor="C1", alpha=0.3, edgecolor="C1", label=r"${\rm NNLO\,MHOU\,(postfit)}$"),
                Patch(facecolor="C2", alpha=0.3, edgecolor="C2", label=r"${\rm NNLO\,MHOU\,(prefit)}$"),
            ]
            ax_main.legend(handles=legend_handles)

            delta_y = 0.1 * (data.central_value.max() + 2 * data.std_error.max() - data.central_value.min())
            ax_main.set_ylim(data.central_value.min() - delta_y, data.central_value.max() + delta_y)
            ax_main.set_xlim(x.min(), x.max())

            ratio = data.central_value / theory.central_value

            ax_ratio.set_ylabel(r"${\rm Ratio}$")
            ax_ratio.set_xlabel(x_label)
            #ax_ratio.set_xscale('log' if not linear else 'linear')

            ratio_min, ratio_max = 1, 1
            for i in range(len(data.central_value)):
                xi = x[i]
                width = x[i + 1] - xi
                cv_data = data.central_value[i]
                se_data = data.std_error[i]
                cv_theory = theory.central_value[i]
                se_theory = theory.std_error[i]
                rect_data = Rectangle((xi, (cv_data - se_data) / cv_data), width, 2.0 * se_data / cv_data,
                                      facecolor="C0", alpha=0.3, edgecolor=None)
                rect_theory = Rectangle((xi, (cv_theory - se_theory) / cv_data), width, 2.0 * se_theory / cv_data,
                                        facecolor="C1", alpha=0.3, edgecolor=None)
                #ratio_min = min(ratio_min, (cv_theory - se_theory) / cv_data, (cv_data - se_data) / cv_data)
                #ratio_max = max(ratio_max, (cv_data + se_data) / cv_data, (cv_theory + se_theory) / cv_data)
                ax_ratio.add_patch(rect_data)
                ax_ratio.add_patch(rect_theory)

            for i in range(len(data.central_value)):
                xi = x[i]
                width = x[i + 1] - xi
                cv_data = data.central_value[i]
                se_data = data.std_error[i]
                cv_theory_prefit = theory_prefit.central_value[i]
                se_theory_prefit = theory_prefit.std_error[i]
                rect_theory_prefit = Rectangle((xi, (cv_theory_prefit - se_theory_prefit) / cv_data), width,
                                               2.0 * se_theory_prefit / cv_data, facecolor="C2", alpha=0.3,
                                               edgecolor=None)
                ratio_min = min(ratio_min, (cv_theory_prefit - se_theory_prefit) / cv_data)
                ratio_max = max(ratio_max, (cv_theory_prefit + se_theory_prefit) / cv_data)
                ax_ratio.add_patch(rect_theory_prefit)

            ax_ratio.axhline(1.0, color='C0', linestyle='-', alpha=1)

            delta_y = 0.1 * (ratio_max - ratio_min)
            #ax_ratio.set_ylim(ratio_min - delta_y, ratio_max + delta_y)
            ax_ratio.set_ylim(0.8, 1.2)
            ax_ratio.set_xlim(x.min(), x.max())

            # python
            # show tick marks at bin edges on the main axis, but keep no labels there
            # Tick positions at unwrapped x
            ax_ratio.set_xticks(x)
            ax_main.set_xticks(x)

            # Tick labels from the original raw x
            tick_labels = [rf"${val:g}$" for val in x_raw]
            ax_ratio.set_xticklabels(tick_labels, rotation=45, ha='right')
            ax_main.set_xticklabels([])  # still hide them on main panel

            # keep existing limits/labels, then tighten layout before saving
            ax_ratio.set_xlim(x.min(), x.max())
            #ax_ratio.set_ylim(0.8, 1.2)
            if metadata.plotting.x_label is not None:
                ax_ratio.set_xlabel(metadata.plotting.x_label)
            else:
                ax_ratio.set_xlabel(x_label_dict[metadata.plotting.plot_x])

            fig.tight_layout()

            observable_name = '_'.join(ds['dataset'].split('_')[1:])
            filename = f"./plots/data-vs-theory/{suffix}_{obs}_{observable_name}.pdf"
            fig.savefig(filename)

