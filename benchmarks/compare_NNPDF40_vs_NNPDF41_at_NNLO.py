# This script compares the NNLO predictions of grids in NNPDF4.0 and NNPDF4.1

# Findings:

# CMS_TTBAR_8TEV_LJ_DIF_YTTBAR gives a 3% mismatch, while ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR gives perfect agreement.
# The CMS grid doesn't carry any metadata, suggesting it was probably ported from appfel grids. The ATLAS grid
# was computed with madgraph using the same dynamical scale choice as used in MATRIX, namely HT/4. This shows our
# 4.1 MATRIX grids are correct: if one MATRIX grid gives agreement with 4.0, also all the other MATRIX grids
# should give agreement with 4.0, because they are all computed from the same run/events

# The CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT grid was computed in NNPDF4.0 with a different dynamical scale, namely
# 1/4 sqrt(ET^t ** 2 + ET^tbar ** 2), which is not the same as HT/4 = 1/ 4 (ET^t + ET^tbar) . This then explains
# why for CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT we are seeing at 13% mismatch: the scale choice is different!

# The 1D and 2D distributions in MATRIX originate from the same events/run so if the 1D is fully benchmarked,
# we can likewise be confident the 2D is correct as well

# bottomline: there are small diferences, but we understand them


import lhapdf
import pineappl
from pineappl.convolutions import Conv, ConvType
import numpy as np

pdfset = "NNPDF40_nnlo_as_01180"
pdf = lhapdf.mkPDF(pdfset, 0)

# ┌───────┬─────┬─────┬─────┬─────┐
# │ index ┆ as  ┆ a   ┆ lf  ┆ lr  │
# │ ---   ┆ --- ┆ --- ┆ --- ┆ --- │
# │ u32   ┆ i64 ┆ i64 ┆ i64 ┆ i64 │
# ╞═══════╪═════╪═════╪═════╪═════╡
# │ 0     ┆ 2   ┆ 0   ┆ 0   ┆ 0   │
# │ 1     ┆ 3   ┆ 0   ┆ 0   ┆ 0   │
# │ 2     ┆ 3   ┆ 0   ┆ 1   ┆ 0   │
# │ 3     ┆ 3   ┆ 0   ┆ 0   ┆ 1   │
# │ 4     ┆ 4   ┆ 0   ┆ 0   ┆ 0   │
# │ 5     ┆ 4   ┆ 0   ┆ 0   ┆ 1   │
# │ 6     ┆ 4   ┆ 0   ┆ 1   ┆ 0   │
# │ 7     ┆ 4   ┆ 0   ┆ 0   ┆ 2   │
# │ 8     ┆ 4   ┆ 0   ┆ 2   ┆ 0   │
# │ 9     ┆ 4   ┆ 0   ┆ 1   ┆ 1   │
# └───────┴─────┴─────┴─────┴─────┘


# we compare for the grids that appear both in NNPDF4.0 and NNPDF4.1, namely:
gridlist = [
    ['ATLASTTBARTOT7TEV-TOPDIFF7TEVTOT', 'ATLAS_TTBAR_7TEV_TOT_X-SEC'],
    ['ATLASTTBARTOT8TEV-TOPDIFF8TEVTOT', 'ATLAS_TTBAR_8TEV_TOT_X-SEC'],
    ['ATLAS_TTBARTOT_13TEV_FULLLUMI-TOPDIFF13TEVTOT', 'ATLAS_TTBAR_13TEV_TOT_X-SEC'],
    ['CMSTTBARTOT7TEV-TOPDIFF7TEVTOT', 'CMS_TTBAR_7TEV_TOT_X-SEC'],
    ['CMSTTBARTOT8TEV-TOPDIFF8TEVTOT', 'CMS_TTBAR_8TEV_TOT_X-SEC'],
    ['CMSTTBARTOT13TEV-TOPDIFF13TEVTOT', 'CMS_TTBAR_13TEV_TOT_X-SEC'],
    ['CMS_TTB_5TEV_TOT', 'CMS_TTBAR_5TEV_TOT_X-SEC'],
    ['CMS_TTB_13TEV_2L_TRAP', 'CMS_TTBAR_13TEV_2L_DIF_YT'],

    # RATIOs
    ['ATLAS_TTB_8TEV_LJ_TRAP', 'ATLAS_TTBAR_8TEV_LJ_DIF_YT'],
    ['ATLAS_TTB_8TEV_LJ_TRAP_TOT', 'ATLAS_TTBAR_8TEV_LJ_DIF_YT-INTEGRATED'],
    ['ATLAS_TTB_8TEV_LJ_TTRAP', 'ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR'],
    ['ATLAS_TTB_8TEV_LJ_TTRAP_TOT', 'ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR-INTEGRATED'],
    ['ATLAS_TTB_8TEV_2L_TTRAP', 'ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR'],
    ['CMSTTBARTOT8TEV-TOPDIFF8TEVTOT', 'ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR-INTEGRATED'],
    ['CMS_TTB_8TEV_2D_TTM_TRAP', 'CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT'],
    ['CMS_TTB_8TEV_2D_TTM_TRAP_TOT', 'CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT-INTEGRATED'],
    ['CMSTOPDIFF8TEVTTRAPNORM-TOPDIFF8TEVTTRAP', 'CMS_TTBAR_8TEV_LJ_DIF_YTTBAR'],
    ['CMSTOPDIFF8TEVTTRAPNORM-TOPDIFF8TEVTOT', 'CMS_TTBAR_8TEV_LJ_DIF_YTTBAR-INTEGRATED'],
]
conv_type = ConvType(polarized=False, time_like=False)
conv_object = Conv(convolution_types=conv_type, pid=2212)

for old_grid_name, new_grid_name in gridlist:

    old_grid = pineappl.grid.Grid.read(
        f"/Users/jaco/Documents/physics_projects/theories_slim/data/grids/4001/{old_grid_name}.pineappl.lz4")

    # 40_009_000 grids are symlinked to 41_000_000 grids, so we can read the same grid name
    new_grid = pineappl.grid.Grid.read(
        f"/Users/jaco/Documents/physics_projects/theories_slim/data/grids/40009000/{new_grid_name}.pineappl.lz4")

    # the total cross section grids are computed with top++ and keep a different order convention
    # convolution in the fit is done correctly, theory prediction entering the fit corresponds to what you get when you do the below
    if "TOT_X-SEC" in new_grid_name:
        orders_nnlo_new = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    else:
        orders_nnlo_new = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)

    new_nnlo_pred = new_grid.convolve(
        pdg_convs=[conv_object, conv_object],  # Similar convolutions for symmetric protons
        xfxs=[pdf.xfxQ2, pdf.xfxQ2],  # Similar PDF sets for symmetric protons
        alphas=pdf.alphasQ2,
        order_mask=orders_nnlo_new
    )

    old_nnlo_pred = old_grid.convolve(
        pdg_convs=[conv_object, conv_object],  # Similar convolutions for symmetric protons
        xfxs=[pdf.xfxQ2, pdf.xfxQ2],  # Similar PDF sets for symmetric protons
        alphas=pdf.alphasQ2,
        order_mask=np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    )



    print(new_grid_name)
    print(new_nnlo_pred/old_nnlo_pred)
