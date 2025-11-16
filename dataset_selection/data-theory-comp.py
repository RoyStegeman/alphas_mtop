from validphys.api import API
from validphys.core import CommonDataSpec
import numpy as np
import nnpdf_data
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
    "MTTBAR-cut	": "251105-jth-dataset-selection-with-ATLAS_TTBAR_MTTBAR-cut",
    "YTTBAR": "251021-jth-dataset-selection-with-ATLAS_TTBAR_YTTBAR",
    #"MTTBAR-PTT": "251021-jth-dataset-selection-with-ATLAS_TTBAR_MTTBAR-PTT",
    #"MTTBAR-YTTBAR": "251021-jth-dataset-selection-with-ATLAS_TTBAR_MTTBAR-YTTBAR",
    "MTTBAR-YTTBAR-cut": "251105-jth-dataset-selection-with-ATLAS_TTBAR_MTTBAR-YTTBAR-cut",
    #"MTTBAR-YT": "251021-jth-dataset-selection-with-ATLAS_TTBAR_MTTBAR-YT",
    #"PTT-YT": "251021-jth-dataset-selection-with-ATLAS_TTBAR_PTT-YT",
}

# CMS
cms_fits = {
    "YT": "251022-jth-dataset-selection-with-CMS_TTBAR_YT",
    "PTT": "251022-jth-dataset-selection-with-CMS_TTBAR_PTT",
    "MTTBAR": "251022-jth-dataset-selection-with-CMS_TTBAR_MTTBAR",
    "MTTBAR-cut": "251105-jth-dataset-selection-with-CMS_TTBAR_MTTBAR-cut",
    "YTTBAR": "251022-jth-dataset-selection-with-CMS_TTBAR_YTTBAR",
    "MTTBAR-PTT": "251022-jth-dataset-selection-with-CMS_TTBAR_MTTBAR-PTT",
    "MTTBAR-YTTBAR": "251022-jth-dataset-selection-with-CMS_TTBAR_MTTBAR-YTTBAR",
    "MTTBAR-YTTBAR-cut": "251105-jth-dataset-selection-with-CMS_TTBAR_MTTBAR-YTTBAR-cut",
    "MTTBAR-YT": "251022-jth-dataset-selection-with-CMS_TTBAR_MTTBAR-YT",
    "PTT-YT": "251022-jth-dataset-selection-with-CMS_TTBAR_PTT-YT",
}

# ATLAS + CMS
atlas_cms_fits = {
    "YT": "251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_YT",
    "PTT": "251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_PTT",
    "MTTBAR": "251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_MTTBAR",
    "MTTBAR-cut": "251105-jth-dataset-selection-with-ATLAS_CMS_TTBAR_MTTBAR-cut",
    "YTTBAR": "251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_YTTBAR",
    "MTTBAR-PTT": "251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_MTTBAR-PTT",
    "MTTBAR-YTTBAR": "251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_MTTBAR-YTTBAR",
    "MTTBAR-YTTBAR-cut": "251105-jth-dataset-selection-with-ATLAS_CMS_TTBAR_MTTBAR-YTTBAR-cut",
    "MTTBAR-YT": "251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_MTTBAR-YT",
    "PTT-YT": "251022-jth-dataset-selection-with-ATLAS_CMS_TTBAR_PTT-YT",
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


for fits in [atlas_fits, cms_fits, atlas_cms_fits]:
    for obs, fit_name in fits.items():



        dict_fit = dict(theoryid=40010006, pdf=fit_name, pdfs=[fit_name], use_cuts="internal", with_shift=False)

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
            metadata = nnpdf_data.load_dataset_metadata(ds["dataset"], variant=None)
            data, theory = API.results(dataset_input=ds, **dict_fit)

            x_label = metadata.plotting.x_label
            y_label = metadata.plotting.y_label
            #title = metadata.plotting.dataset_label

            #ax_main.set_title(title)
            ax_main.set_title(label_dict[ds["dataset"].split("_DIF")[0]])
            ax_main.set_ylabel(y_label)
            ax_main.set_xticklabels([])
            ax_main.grid(True)




            n_data = len(data)
            x = np.arange(n_data)

            for xi, cv, se in zip(x, data.central_value, data.std_error):
                rect = Rectangle((xi, cv - se), 1, 2.0 * se, facecolor="C0", alpha=0.3, edgecolor="C0")
                ax_main.add_patch(rect)
            for xi, cv, se in zip(x, theory.central_value, theory.std_error):
                rect = Rectangle((xi, cv - se), 1, 2.0 * se, facecolor="C1", alpha=0.3, edgecolor="C1")
                ax_main.add_patch(rect)

            ax_main.step(x, data.central_value, where='post', color="C0")
            ax_main.step(x, theory.central_value, where='post', color="C1")

            legend_handles = [
                Patch(facecolor="C0", alpha=0.15, edgecolor="C0", label=r"${\rm Data}$"),
                Patch(facecolor="C1", alpha=0.15, edgecolor="C1", label=r"${\rm Theory}$"),
            ]
            ax_main.legend(handles=legend_handles)

            delta_y = 0.1 * (data.central_value.max() + 2 * data.std_error.max() - data.central_value.min())
            ax_main.set_ylim(data.central_value.min() - delta_y, data.central_value.max() + delta_y)
            ax_main.set_xlim(x.min(), x.max())

            ratio = data.central_value / theory.central_value

            ax_ratio.set_ylabel(r"${\rm Ratio}$")
            ax_ratio.set_xlabel(x_label)
            ax_ratio.grid(True)
            ratio_min, ratio_max = 1, 1
            for xi, cv_data, se_data, cv_theory, se_theory in zip(x, data.central_value, data.std_error, theory.central_value, theory.std_error):
                rect_data = Rectangle((xi, (cv_data - se_data) / cv_data), 1, 2.0 * se_data / cv_data, facecolor="C0", alpha=0.3, edgecolor="C0")
                rect_theory = Rectangle((xi, (cv_theory - se_theory) / cv_data), 1, 2.0 * se_theory / cv_data, facecolor="C1", alpha=0.3, edgecolor="C1")
                ratio_min = min(ratio_min, (cv_theory - se_theory) / cv_data, (cv_data - se_data) / cv_data)
                ratio_max = max(ratio_max, (cv_data + se_data) / cv_data, (cv_theory + se_theory) / cv_data)
                ax_ratio.add_patch(rect_data)
                ax_ratio.add_patch(rect_theory)

            delta_y = 0.1 * (ratio_max - ratio_min)
            ax_ratio.set_ylim(ratio_min - delta_y, ratio_max + delta_y)
            ax_ratio.set_xlim(x.min(), x.max())



            fig.savefig(f"./plots/data-vs-theory/{ds["dataset"]}.pdf")

