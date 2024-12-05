# -*- coding: utf-8 -*-
import pathlib
import subprocess
import pineappl
from pineappl.grid import Grid
import pandas as pd
import numpy as np


# Define directories
output_dir = pathlib.Path(".")
matrix_result_dir = pathlib.Path("../../MATRIX/result")
pineappl_dir = pathlib.Path("NNLO-run/PineAPPL_grids")

# Define Matrix theory details
theory = "40006001"
output_dir = output_dir / theory  # Ensure output directory is scoped to the theory
output_dir.mkdir(parents=True, exist_ok=True)
# temp_dir = output_dir / "temp"
# temp_dir.mkdir(parents=True, exist_ok=True)

def get_input_dir(matrix_run: str) -> pathlib.Path:
    """Construct the full input directory path."""
    return matrix_result_dir / matrix_run / theory / pineappl_dir

def rename_grid(input_grid, output_grid):
    """
    Rename a grid file by copying it to a new location.

    Parameters
    ----------
    input_grid : str or pathlib.Path
        The path to the input grid file.
    output_grid : str or pathlib.Path
        The path to the output grid file.

    Returns
    -------
    None
    """

    try:
        input_dir = get_input_dir(matrix_run)
        subprocess.run(["cp", input_dir / input_grid, output_dir / output_grid], check=True)
        print(f"Renamed: {input_dir / input_grid} -> {output_dir / output_grid}")
    except subprocess.CalledProcessError as e:
        print(f"Error copying file: {e}")
        #raise

def merge_bins(input_grid, output_grid):
    """
    Merge bins in a grid using PineAPPL.

    Parameters
    ----------
    input_grid : str or pathlib.Path
        Path to the input grid file.
    output_grid : str or pathlib.Path
        Path to the output grid file.

    Returns
    -------
    str or pathlib.Path
        Path to the output grid file with merged bins.
    """

    grid = Grid.read(f"{input_grid}")
    n_bins = grid.bins()

    try:
        subprocess.run(
            [
                "pineappl",
                "write",
                "--merge-bins",
                f"0-{n_bins-1}",
                input_grid,
                output_grid
            ],
            check=True,
        )
        return output_grid
    except subprocess.CalledProcessError as e:
        print(f"Error while merging bins {e}")

def integrate_1d(matrix_run, input_grid, output_grid):
    """
    Integrate 1D distribution from a PineAPPL grid.

    Parameters
    ----------
    matrix_run : str
        The name of the MATRIX run.
    input_grid : str or pathlib.Path
        The path to the input grid file.
    output_grid : str or pathlib.Path
        The path to the output grid file.

    Returns
    -------
    None
    """

    input_dir = get_input_dir(matrix_run)
    merged_grid = merge_bins(input_dir / input_grid, output_dir / f"temp_{output_grid}")
    grid = Grid.read(f"{merged_grid}")
    bin_dims = grid.bin_dimensions()

    bin_limits = [
        (left, right)
        for left, right in zip(grid.bin_left(bin_dims - 1), grid.bin_right(bin_dims - 1))
    ]

    normalizations = [1.0 for _ in grid.bin_normalizations()]

    remapper = pineappl.bin.BinRemapper(np.array(normalizations), bin_limits)

    # Modify the bin normalization
    grid.set_remapper(remapper)

    # Save the modified grid
    grid.write_lz4(output_dir / output_grid)
    subprocess.run(["rm", output_dir / f"temp_{output_grid}"], check=True)

    print(f"Successfully integrated : {input_grid} -> {output_grid}")

matrix_run = "run_ATLAS_TTBAR_13TEV_HADR_DIF"

integrate_1d(matrix_run, "m_ttx_NNLO.QCD.pineappl.lz4", "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR_INTEGRATED.pineappl.lz4" )
rename_grid("m_ttx_NNLO.QCD.pineappl.lz4", "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR.pineappl.lz4")

integrate_1d(
    matrix_run,
    "m_ttx_NNLO.QCD.pineappl.lz4",
    "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR_INTEGRATED.pineappl.lz4"
)

rename_grid("y_ttx_NNLO.QCD.pineappl.lz4", "ATLAS_TTBAR_13TEV_HADR_DIF_YTTBAR.pineappl.lz4")
integrate_1d(matrix_run,
    "y_ttx_NNLO.QCD.pineappl.lz4",
    "ATLAS_TTBAR_13TEV_HADR_DIF_YTTBAR_INTEGRATED.pineappl.lz4"
)

matrix_run = "run_ATLAS_TTBAR_13TEV_LJ_DIF"
input_dir = get_input_dir(matrix_run)

# ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR
rename_grid("atlas_mttbar_NNLO.QCD.pineappl.lz4", "ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR.pineappl.lz4")

# ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-INTEGRATED
integrate_1d(matrix_run,
    "atlas_mttbar_NNLO.QCD.pineappl.lz4",
    "ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-INTEGRATED.pineappl.lz4"
)

# ATLAS_TTBAR_13TEV_LJ_DIF_PTT
rename_grid("atlas_pTt_NNLO.QCD.pineappl.lz4", "ATLAS_TTBAR_13TEV_LJ_DIF_PTT.pineappl.lz4")

# ATLAS_TTBAR_13TEV_LJ_DIF_PTT-INTEGRATED
integrate_1d(matrix_run,
    "atlas_pTt_NNLO.QCD.pineappl.lz4",
    "ATLAS_TTBAR_13TEV_LJ_DIF_PTT-INTEGRATED.pineappl.lz4"
)

# ATLAS_TTBAR_13TEV_LJ_DIF_YT
rename_grid("atlas_yt_NNLO.QCD.pineappl.lz4", "ATLAS_TTBAR_13TEV_LJ_DIF_YT.pineappl.lz4")

# ATLAS_TTBAR_13TEV_LJ_DIF_YT-INTEGRATED
integrate_1d(matrix_run,
    "atlas_yt_NNLO.QCD.pineappl.lz4",
    "ATLAS_TTBAR_13TEV_LJ_DIF_YT-INTEGRATED.pineappl.lz4"
)

# ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR
rename_grid("atlas_yttbar_NNLO.QCD.pineappl.lz4", "ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR.pineappl.lz4")

# ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR-INTEGRATED
integrate_1d(matrix_run,
    "atlas_yttbar_NNLO.QCD.pineappl.lz4",
    "ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR-INTEGRATED.pineappl.lz4"
)

# ATLAS_TTBAR_13TEV_TOT_X-SEC
rename_grid("total_rate_NNLO.QCD.pineappl.lz4", "ATLAS_TTBAR_13TEV_TOT_X-SEC.pineappl.lz4")

# CMS_TTBAR_13TEV_2L_DIF_MTTBAR
rename_grid("cms_2l_mttbar_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_2L_DIF_MTTBAR.pineappl.lz4")

# CMS_TTBAR_13TEV_2L_DIF_MTTBAR-INTEGRATED
integrate_1d(matrix_run,
    "cms_2l_mttbar_NNLO.QCD.pineappl.lz4",
    "CMS_TTBAR_13TEV_2L_DIF_MTTBAR-INTEGRATED.pineappl.lz4"
)

# CMS_TTBAR_13TEV_2L_DIF_PTT
rename_grid("cms_2l_pTt_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_2L_DIF_PTT.pineappl.lz4")

# CMS_TTBAR_13TEV_2L_DIF_PTT-INTEGRATED
integrate_1d(matrix_run,
    "cms_2l_pTt_NNLO.QCD.pineappl.lz4",
    "CMS_TTBAR_13TEV_2L_DIF_PTT-INTEGRATED.pineappl.lz4"
)

# CMS_TTBAR_13TEV_2L_DIF_YT
rename_grid("cms_2l_yt_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_2L_DIF_YT.pineappl.lz4")

# CMS_TTBAR_13TEV_2L_DIF_YT-INTEGRATED
integrate_1d(matrix_run,
    "cms_2l_yt_NNLO.QCD.pineappl.lz4",
    "CMS_TTBAR_13TEV_2L_DIF_YT-INTEGRATED.pineappl.lz4"
)

# CMS_TTBAR_13TEV_2L_DIF_YTTBAR
rename_grid("cms_2l_yttbar_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_2L_DIF_YTTBAR.pineappl.lz4")

# CMS_TTBAR_13TEV_2L_DIF_YTTBAR-INTEGRATED
integrate_1d(matrix_run,
    "cms_2l_yttbar_NNLO.QCD.pineappl.lz4",
    "CMS_TTBAR_13TEV_2L_DIF_YTTBAR-INTEGRATED.pineappl.lz4"
)

# CMS_TTBAR_13TEV_LJ_DIF_MTTBAR
rename_grid("cms_lj_mttbar_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_LJ_DIF_MTTBAR.pineappl.lz4")

# CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-INTEGRATED
integrate_1d(matrix_run,
    "cms_lj_mttbar_NNLO.QCD.pineappl.lz4",
    "CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-INTEGRATED.pineappl.lz4"
)

# CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-YTTBAR
rename_grid("cms_lj_mttbar-yttbar_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-YTTBAR.pineappl.lz4")

# CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-YTTBAR-INTEGRATED
# TODO: how to integrate 2D dist

# CMS_TTBAR_13TEV_LJ_DIF_PTT
rename_grid("cms_lj_pTt_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_LJ_DIF_PTT.pineappl.lz4")


# CMS_TTBAR_13TEV_LJ_DIF_PTT-INTEGRATED
integrate_1d(matrix_run,
    "cms_lj_pTt_NNLO.QCD.pineappl.lz4",
    "CMS_TTBAR_13TEV_LJ_DIF_PTT-INTEGRATED.pineappl.lz4"
)

# CMS_TTBAR_13TEV_LJ_DIF_YT
rename_grid("cms_lj_yt_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_LJ_DIF_YT.pineappl.lz4")

# CMS_TTBAR_13TEV_LJ_DIF_YT-INTEGRATED
integrate_1d(matrix_run,
    "cms_lj_yt_NNLO.QCD.pineappl.lz4",
    "CMS_TTBAR_13TEV_LJ_DIF_YT-INTEGRATED.pineappl.lz4"
)

# CMS_TTBAR_13TEV_LJ_DIF_YTTBAR
rename_grid("cms_lj_yttbar_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_LJ_DIF_YTTBAR.pineappl.lz4")

# CMS_TTBAR_13TEV_LJ_DIF_YTTBAR-INTEGRATED
integrate_1d(matrix_run,
    "cms_lj_yttbar_NNLO.QCD.pineappl.lz4",
    "CMS_TTBAR_13TEV_LJ_DIF_YTTBAR-INTEGRATED.pineappl.lz4"
)

# CMS_TTBAR_13TEV_TOT_X-SEC
rename_grid("total_rate_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_TOT_X-SEC.pineappl.lz4")


matrix_run = "run_ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR"
# input_dir = get_input_dir(matrix_run)
#
# run_pineappl_merge(
#     "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR.pineappl.lz4",
#     "dd1_NNLO.QCD.pineappl.lz4",
#     "dd2_NNLO.QCD.pineappl.lz4",
#     "dd3_NNLO.QCD.pineappl.lz4"
# )

# TODO: how to integrate 2D dist?

# for i, n_bins in {1: 4, 2:4, 3:3}.items():
#     integrate_1d(matrix_run,
#         f"dd{i}_NNLO.QCD.pineappl.lz4",
#         temp_dir / f"dd{i}_NNLO.QCD_INTEGRATED.pineappl.lz4",
#         f"0-{n_bins-1}"
#     )
#
# run_pineappl_merge("ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR_INTEGRATED.pineappl.lz4", *list(temp_dir.glob("*.lz4")))
# integrate_1d(matrix_run,
#     "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR_INTEGRATED.pineappl.lz4",
#     "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR_INTEGRATED_test.pineappl.lz4",
#     "0-2"
# )

for exp in ["atlas", "cms"]:

    matrix_run = "run_ATLAS_TTBAR_8TEV_DIF"
    input_dir = get_input_dir(matrix_run)

    rename_grid(f"{exp}_2l_mttbar_NNLO.QCD.pineappl.lz4", f"{exp.upper()}_TTBAR_8TEV_2L_DIF_MTTBAR.pineappl.lz4")
    integrate_1d(matrix_run,
        f"{exp}_2l_mttbar_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_2L_DIF_MTTBAR_INTEGRATED.pineappl.lz4"
    )

    rename_grid(f"{exp}_2l_yttbar_NNLO.QCD.pineappl.lz4", f"{exp.upper()}_TTBAR_8TEV_2L_DIF_YTTBAR.pineappl.lz4")
    integrate_1d(matrix_run,
        f"{exp}_2l_yttbar_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_2L_DIF_YTTBAR_INTEGRATED.pineappl.lz4"
    )

    rename_grid(f"{exp}_lj_mttbar_NNLO.QCD.pineappl.lz4", f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_MTTBAR.pineappl.lz4")
    integrate_1d(matrix_run,
        f"{exp}_lj_mttbar_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_MTTBAR_INTEGRATED.pineappl.lz4"
    )

    rename_grid(f"{exp}_lj_pTt_NNLO.QCD.pineappl.lz4", f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_PTT.pineappl.lz4")
    integrate_1d(matrix_run,
        f"{exp}_lj_pTt_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_PTT_INTEGRATED.pineappl.lz4"
    )

    rename_grid(f"{exp}_lj_yt_NNLO.QCD.pineappl.lz4", f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_YT.pineappl.lz4")
    integrate_1d(matrix_run,
        f"{exp}_lj_yt_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_YT_INTEGRATED.pineappl.lz4"
    )

    rename_grid(f"{exp}_lj_yttbar_NNLO.QCD.pineappl.lz4", f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_YTTBAR.pineappl.lz4")
    integrate_1d(matrix_run,
        f"{exp}_lj_yttbar_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_YTTBAR_INTEGRATED.pineappl.lz4"
    )

    rename_grid("total_rate_NNLO.QCD.pineappl.lz4", f"{exp.upper()}_TTBAR_8TEV_TOT_X-SEC.pineappl.lz4")

for exp in ["atlas", "cms"]:

    matrix_run = "run_ATLAS_TTBAR_7TEV_TOT_X-SEC"
    input_dir = get_input_dir(matrix_run)
    rename_grid("total_rate_NNLO.QCD.pineappl.lz4", f"{exp.upper()}_TTBAR_7TEV_TOT_X-SEC.pineappl.lz4")

matrix_run = "run_CMS_TTBAR_5TEV_TOT_X-SEC"
input_dir = get_input_dir(matrix_run)
rename_grid("total_rate_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_5TEV_TOT_X-SEC.pineappl.lz4")

matrix_run = "run_CMS_TTBAR_8TEV_2L_DIF"
input_dir = get_input_dir(matrix_run)

# CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT
rename_grid("mttbar-yt_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT.pineappl.lz4")

# CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT-INTEGRATED

# CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YTTBAR-INTEGRATED

# CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YTTBAR
rename_grid("mttbar-yttbar_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YTTBAR.pineappl.lz4")

# CMS_TTBAR_8TEV_2L_DIF_PTT-YT-INTEGRATED

# CMS_TTBAR_8TEV_2L_DIF_PTT-YT
rename_grid("pTt-yt_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_8TEV_2L_DIF_PTT-YT .pineappl.lz4")