# This script compares two independent MATRIX runs of the same process at various orders in perturbation theory
# It shows the total error is driven by the NNLO contribution

import lhapdf
import pineappl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
from scipy import stats

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 16})
rc('text', usetex=True)

pdfset = "NNPDF40_nnlo_as_01180"
pdf = lhapdf.mkPDF(pdfset, 0)


grid_tanishq = pineappl.grid.Grid.read("../baseline_tanishq/ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR.pineappl.lz4")
predictions_tanishq = grid_tanishq.convolve_with_one(2212, pdf.xfxQ2, pdf.alphasQ2)

grid_jaco = pineappl.grid.Grid.read("../data/grids_reproducing_tanishq/40006000/ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR.pineappl.lz4")
predictions_jaco = grid_jaco.convolve_with_one(2212, pdf.xfxQ2, pdf.alphasQ2) * 1e-3


df_tanishq_nnlo = pd.DataFrame(
    {
        "bins": range(predictions_tanishq.size),
        "predictions": predictions_tanishq,
    }
)


df_jaco_nnlo = pd.DataFrame(
    {
        "bins": range(predictions_jaco.size),
        "predictions": predictions_jaco,
    }
)

predictions_tanishq_nlo = grid_tanishq.convolve_with_one(2212, pdf.xfxQ2, pdf.alphasQ2, order_mask=[True, True, False, False, False, False, False, False, False])
predictions_jaco_nlo = grid_jaco.convolve_with_one(2212, pdf.xfxQ2, pdf.alphasQ2, order_mask=[True, True, False, False, False, False, False, False, False]) * 1e-3

df_tanishq_nlo = pd.DataFrame(
    {
        "bins": range(predictions_tanishq_nlo.size),
        "predictions": predictions_tanishq_nlo,
    }
)

df_jaco_nlo = pd.DataFrame(
    {
        "bins": range(predictions_jaco_nlo.size),
        "predictions": predictions_jaco_nlo,
    }
)

predictions_tanishq_lo = grid_tanishq.convolve_with_one(2212, pdf.xfxQ2, pdf.alphasQ2, order_mask=[True, False, False, False, False, False, False, False, False])
predictions_jaco_lo = grid_jaco.convolve_with_one(2212, pdf.xfxQ2, pdf.alphasQ2, order_mask=[True, False, False, False, False, False, False, False, False]) * 1e-3

df_tanishq_lo = pd.DataFrame(
    {
        "bins": range(predictions_tanishq_lo.size),
        "predictions": predictions_tanishq_lo,
    }
)

df_jaco_lo = pd.DataFrame(
    {
        "bins": range(predictions_jaco_lo.size),
        "predictions": predictions_jaco_lo,
    }
)


def plot_combined(df_nnlo_1, df_nnlo_2, df_nlo_1, df_nlo_2, df_lo_1, df_lo_2):
    fig, axes = plt.subplots(6, 1, figsize=(10, 24), gridspec_kw={'height_ratios': [3, 1, 3, 1, 3, 1], })

    binning = np.array([325.0, 458.75, 565.75, 646.0, 753.0, 860.0, 967.0, 1100.75, 1261.25, 3000.0])
    x_data = binning[:-1] + 0.5 * (binning[1:] - binning[:-1])
    mc_error = np.array([21.7479, 6.91872, 8.15598, 3.10534, 1.25490, 1.67165, 0.593426, 0.489423, 0.0198982]) * 1e-3

    titles = ["NNLO", "NLO", "LO"]
    df_pairs = [(df_nnlo_1, df_nnlo_2), (df_nlo_1, df_nlo_2), (df_lo_1, df_lo_2)]

    for i, ((df_1, df_2), title) in enumerate(zip(df_pairs, titles)):
        ax = axes[i * 2]
        # remove tick labels
        ax.set_xticklabels([])
        ax_ratio = axes[i * 2 + 1]
        ax.set_title(f"ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR ({title})")
        ax.set_yscale('log')

        if i == 0:
            ax.errorbar(x_data, df_1['predictions'].values, mc_error, lw=1, fmt='o', color='k', capsize=3, label='Jaco')
            ax.errorbar(x_data, df_2['predictions'].values, mc_error, lw=1, fmt='o', color='r', capsize=3,
                        label='Tanishq')

            ndat = len(df_2['predictions'].values)
            chi2_tot = np.sum((df_1['predictions'].values - df_2['predictions'].values) ** 2 / (2 * mc_error ** 2))
            p_value = 1 - stats.chi2.cdf(chi2_tot, ndat)
            ax.text(0.95, 0.75, r"$\chi^2/n_{{\rm dat}}={:.2f}$".format(chi2_tot / ndat),
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes)
            ax.text(0.95, 0.65, r"${{\rm p-value}}={:.2f}$".format(p_value),
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes)

        else:
            ax.scatter(x_data, df_1['predictions'].values, label='Jaco', color='k')
            ax.scatter(x_data, df_2['predictions'].values, label='Tanishq', color='r')

        ax.legend()

        # Ratio plot
        ratio = df_1['predictions'].values / df_2['predictions'].values
        if i == 0:
            ax_ratio.errorbar(x_data, df_1['predictions'].values / df_2['predictions'].values,
                            mc_error / df_2['predictions'].values, fmt='o', lw=1, color='k', capsize=3)
            ax_ratio.errorbar(x_data, df_2['predictions'].values / df_2['predictions'].values,
                            mc_error / df_2['predictions'].values, fmt='o', lw=1, color='r', capsize=3)
            ax_ratio.set_ylim(0.96, 1.04)
        else:
            ax_ratio.scatter(x_data, ratio, color='k')
            deltay_ratio = 1.1 * np.max(np.abs(ratio - 1))
            ax_ratio.set_ylim(1 - deltay_ratio, 1 + deltay_ratio)
        ax_ratio.axhline(1, color='k', linestyle='dashed')
        ax_ratio.set_ylabel("Ratio")

        ax_ratio.set_xlabel(r'$m_{t\bar{t}}\;\mathrm{[GeV]}$')

    axes[2].set_ylabel(r'$d\sigma/dm_{t\bar{t}}$')

    plt.tight_layout()
    return fig


fig_combined = plot_combined(df_jaco_nnlo, df_tanishq_nnlo, df_jaco_nlo, df_tanishq_nlo, df_jaco_lo, df_tanishq_lo)
fig_combined.savefig("comparison_all_orders.pdf")


