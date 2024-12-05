# -*- coding: utf-8 -*-
import pathlib
import subprocess
import pandas as pd
import numpy as np
import pineappl
from pineappl.grid import Grid

theory_id = "40006001"


class MatrixRun:
    def __init__(self, matrix_run, theory_id):

        self.matrix_run = matrix_run
        self.theory_id = theory_id
        self.matrix_result_dir = pathlib.Path("../../MATRIX/result")
        self.pineappl_dir = pathlib.Path("NNLO-run/PineAPPL_grids")
        self.output_dir = pathlib.Path(self.theory_id)
        self.input_dir = self.get_input_dir()

    def get_input_dir(self) -> pathlib.Path:
        """Construct the full input directory path."""
        return (
            self.matrix_result_dir
            / self.matrix_run
            / self.theory_id
            / self.pineappl_dir
        )

    def rename_grid(self, input_grid, output_grid):
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
            subprocess.run(
                ["cp", self.input_dir / input_grid, self.output_dir / output_grid],
                check=True,
            )
            print(
                f"Renamed: {self.input_dir / input_grid} -> {self.output_dir / output_grid}"
            )
        except subprocess.CalledProcessError as e:
            print(f"Error copying file: {e}")

    def merge_bins(self, input_grid, output_grid):
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
                    f"0-{n_bins - 1}",
                    input_grid,
                    output_grid,
                ],
                check=True,
            )
            return output_grid
        except subprocess.CalledProcessError as e:
            print(f"Error while merging bins {e}")

    def combine_grids(self, output_grid, input_grids):

        input = [self.input_dir / grid for grid in input_grids]
        subprocess.run(["pineappl", "merge", self.output_dir / output_grid, *input], check=True)

    def integrate_1d(self, input_grid, output_grid):
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

        merged_grid = self.merge_bins(self.input_dir / input_grid, self.output_dir / f"temp_{output_grid}")
        grid = Grid.read(f"{merged_grid}")
        bin_dims = grid.bin_dimensions()

        bin_limits = [
            (left, right)
            for left, right in zip(
                grid.bin_left(bin_dims - 1), grid.bin_right(bin_dims - 1)
            )
        ]

        normalizations = [1.0 for _ in grid.bin_normalizations()]

        remapper = pineappl.bin.BinRemapper(np.array(normalizations), bin_limits)

        # Modify the bin normalization
        grid.set_remapper(remapper)

        # Save the modified grid
        grid.write_lz4(self.output_dir / output_grid)
        subprocess.run(["rm", self.output_dir / f"temp_{output_grid}"], check=True)

        print(f"Successfully integrated : {input_grid} -> {output_grid}")

    def find_ranges(self, bins):

        # Find where the bin value changes
        change_indices = np.where(np.diff(bins) != 0)[0]

        # Add start and end indices
        ranges = np.split(np.arange(len(bins)), change_indices + 1)

        return [(r[0], r[-1]) for r in ranges]


    def integrate_2d(self, input_grid, output_grid):



        grid = Grid.read(self.output_dir / f"{input_grid}")

        bin_ranges = self.find_ranges(grid.bin_left(0))

        input = self.output_dir / f"{input_grid}"
        for i, range in enumerate(bin_ranges):
            output = self.output_dir / f"temp_{i}_{output_grid}"
            n_bins = range[1] - range[0] + 1
            subprocess.run(["pineappl", "write", "--merge-bins", f"{i}-{i + n_bins - 1}", input,
                            output], check=True)
            input = output

        import pdb;
        pdb.set_trace()


        for i in range(0, grid.bins()):
            grid.bin_left(0)[i] = 0.0
            grid.bin_right(0)[i] = 1.0
            grid.bin_left(1)[i] = 0.0
            grid.bin_right(1)[i] = 1.0

        bin_dims = grid.bin_dimensions()
        df = pd.DataFrame({})
        for bin_dim in range(bin_dims):
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            f"dim {bin_dim} left": grid.bin_left(bin_dim),
                            f"dim {bin_dim} right": grid.bin_right(bin_dim),
                        }
                    ),
                ],
                axis=0,
            )
        import pdb;
        pdb.set_trace()

        # for i, n_bins in {1: 4, 2:4, 3:3}.items():
        #     self.integrate_1d(
        #         f"dd{i}_NNLO.QCD.pineappl.lz4",
        #         temp_dir / f"dd{i}_NNLO.QCD_INTEGRATED.pineappl.lz4",
        #         f"0-{n_bins-1}"
        #     )
        #
        # run_pineappl_merge("ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR_INTEGRATED.pineappl.lz4", *list(temp_dir.glob("*.lz4")))
        # matrix_run.integrate_1d(
        #     "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR_INTEGRATED.pineappl.lz4",
        #     "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR_INTEGRATED_test.pineappl.lz4",
        #     "0-2"
        # )
        #
        # self.integrate_1d()


# matrix_run = MatrixRun("run_ATLAS_TTBAR_13TEV_HADR_DIF", theory_id=theory_id)
# matrix_run.integrate_1d(
#     "m_ttx_NNLO.QCD.pineappl.lz4",
#     "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR_INTEGRATED.pineappl.lz4",
# )
# matrix_run.rename_grid(
#     "m_ttx_NNLO.QCD.pineappl.lz4", "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR.pineappl.lz4"
# )
#
# matrix_run.integrate_1d(
#     "m_ttx_NNLO.QCD.pineappl.lz4",
#     "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR_INTEGRATED.pineappl.lz4",
# )
#
# matrix_run.rename_grid(
#     "y_ttx_NNLO.QCD.pineappl.lz4", "ATLAS_TTBAR_13TEV_HADR_DIF_YTTBAR.pineappl.lz4"
# )
# matrix_run.integrate_1d(
#     "y_ttx_NNLO.QCD.pineappl.lz4",
#     "ATLAS_TTBAR_13TEV_HADR_DIF_YTTBAR_INTEGRATED.pineappl.lz4",
# )
#
# matrix_run = MatrixRun("run_ATLAS_TTBAR_13TEV_LJ_DIF", theory_id=theory_id)
#
# # ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR
# matrix_run.rename_grid(
#     "atlas_mttbar_NNLO.QCD.pineappl.lz4", "ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR.pineappl.lz4"
# )
#
# # ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-INTEGRATED
# matrix_run.integrate_1d(
#     "atlas_mttbar_NNLO.QCD.pineappl.lz4",
#     "ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-INTEGRATED.pineappl.lz4",
# )
#
# # ATLAS_TTBAR_13TEV_LJ_DIF_PTT
# matrix_run.rename_grid(
#     "atlas_pTt_NNLO.QCD.pineappl.lz4", "ATLAS_TTBAR_13TEV_LJ_DIF_PTT.pineappl.lz4"
# )
#
# # ATLAS_TTBAR_13TEV_LJ_DIF_PTT-INTEGRATED
# matrix_run.integrate_1d(
#     "atlas_pTt_NNLO.QCD.pineappl.lz4",
#     "ATLAS_TTBAR_13TEV_LJ_DIF_PTT-INTEGRATED.pineappl.lz4",
# )
#
# # ATLAS_TTBAR_13TEV_LJ_DIF_YT
# matrix_run.rename_grid(
#     "atlas_yt_NNLO.QCD.pineappl.lz4", "ATLAS_TTBAR_13TEV_LJ_DIF_YT.pineappl.lz4"
# )
#
# # ATLAS_TTBAR_13TEV_LJ_DIF_YT-INTEGRATED
# matrix_run.integrate_1d(
#     "atlas_yt_NNLO.QCD.pineappl.lz4",
#     "ATLAS_TTBAR_13TEV_LJ_DIF_YT-INTEGRATED.pineappl.lz4",
# )
#
# # ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR
# matrix_run.rename_grid(
#     "atlas_yttbar_NNLO.QCD.pineappl.lz4", "ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR.pineappl.lz4"
# )
#
# # ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR-INTEGRATED
# matrix_run.integrate_1d(
#     "atlas_yttbar_NNLO.QCD.pineappl.lz4",
#     "ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR-INTEGRATED.pineappl.lz4",
# )
#
# # ATLAS_TTBAR_13TEV_TOT_X-SEC
# matrix_run.rename_grid(
#     "total_rate_NNLO.QCD.pineappl.lz4", "ATLAS_TTBAR_13TEV_TOT_X-SEC.pineappl.lz4"
# )
#
# # CMS_TTBAR_13TEV_2L_DIF_MTTBAR
# matrix_run.rename_grid(
#     "cms_2l_mttbar_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_2L_DIF_MTTBAR.pineappl.lz4"
# )
#
# # CMS_TTBAR_13TEV_2L_DIF_MTTBAR-INTEGRATED
# matrix_run.integrate_1d(
#     "cms_2l_mttbar_NNLO.QCD.pineappl.lz4",
#     "CMS_TTBAR_13TEV_2L_DIF_MTTBAR-INTEGRATED.pineappl.lz4",
# )
#
# # CMS_TTBAR_13TEV_2L_DIF_PTT
# matrix_run.rename_grid(
#     "cms_2l_pTt_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_2L_DIF_PTT.pineappl.lz4"
# )
#
# # CMS_TTBAR_13TEV_2L_DIF_PTT-INTEGRATED
# matrix_run.integrate_1d(
#     "cms_2l_pTt_NNLO.QCD.pineappl.lz4",
#     "CMS_TTBAR_13TEV_2L_DIF_PTT-INTEGRATED.pineappl.lz4",
# )
#
# # CMS_TTBAR_13TEV_2L_DIF_YT
# matrix_run.rename_grid(
#     "cms_2l_yt_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_2L_DIF_YT.pineappl.lz4"
# )
#
# # CMS_TTBAR_13TEV_2L_DIF_YT-INTEGRATED
# matrix_run.integrate_1d(
#     "cms_2l_yt_NNLO.QCD.pineappl.lz4",
#     "CMS_TTBAR_13TEV_2L_DIF_YT-INTEGRATED.pineappl.lz4",
# )
#
# # CMS_TTBAR_13TEV_2L_DIF_YTTBAR
# matrix_run.rename_grid(
#     "cms_2l_yttbar_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_2L_DIF_YTTBAR.pineappl.lz4"
# )
#
# # CMS_TTBAR_13TEV_2L_DIF_YTTBAR-INTEGRATED
# matrix_run.integrate_1d(
#     "cms_2l_yttbar_NNLO.QCD.pineappl.lz4",
#     "CMS_TTBAR_13TEV_2L_DIF_YTTBAR-INTEGRATED.pineappl.lz4",
# )
#
# # CMS_TTBAR_13TEV_LJ_DIF_MTTBAR
# matrix_run.rename_grid(
#     "cms_lj_mttbar_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_LJ_DIF_MTTBAR.pineappl.lz4"
# )
#
# # CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-INTEGRATED
# matrix_run.integrate_1d(
#     "cms_lj_mttbar_NNLO.QCD.pineappl.lz4",
#     "CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-INTEGRATED.pineappl.lz4",
# )
#
# # CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-YTTBAR
# matrix_run.rename_grid(
#     "cms_lj_mttbar-yttbar_NNLO.QCD.pineappl.lz4",
#     "CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-YTTBAR.pineappl.lz4",
# )
#
# # CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-YTTBAR-INTEGRATED
# # TODO: how to integrate 2D dist
#
# # CMS_TTBAR_13TEV_LJ_DIF_PTT
# matrix_run.rename_grid(
#     "cms_lj_pTt_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_LJ_DIF_PTT.pineappl.lz4"
# )
#
#
# # CMS_TTBAR_13TEV_LJ_DIF_PTT-INTEGRATED
# matrix_run.integrate_1d(
#     "cms_lj_pTt_NNLO.QCD.pineappl.lz4",
#     "CMS_TTBAR_13TEV_LJ_DIF_PTT-INTEGRATED.pineappl.lz4",
# )
#
# # CMS_TTBAR_13TEV_LJ_DIF_YT
# matrix_run.rename_grid(
#     "cms_lj_yt_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_LJ_DIF_YT.pineappl.lz4"
# )
#
# # CMS_TTBAR_13TEV_LJ_DIF_YT-INTEGRATED
# matrix_run.integrate_1d(
#     "cms_lj_yt_NNLO.QCD.pineappl.lz4",
#     "CMS_TTBAR_13TEV_LJ_DIF_YT-INTEGRATED.pineappl.lz4",
# )
#
# # CMS_TTBAR_13TEV_LJ_DIF_YTTBAR
# matrix_run.rename_grid(
#     "cms_lj_yttbar_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_LJ_DIF_YTTBAR.pineappl.lz4"
# )
#
# # CMS_TTBAR_13TEV_LJ_DIF_YTTBAR-INTEGRATED
# matrix_run.integrate_1d(
#     "cms_lj_yttbar_NNLO.QCD.pineappl.lz4",
#     "CMS_TTBAR_13TEV_LJ_DIF_YTTBAR-INTEGRATED.pineappl.lz4",
# )
#
# # CMS_TTBAR_13TEV_TOT_X-SEC
# matrix_run.rename_grid(
#     "total_rate_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_13TEV_TOT_X-SEC.pineappl.lz4"
# )


matrix_run = MatrixRun(
    "run_ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR", theory_id=theory_id
)

matrix_run.combine_grids("ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR.pineappl.lz4",
        input_grids=["dd1_NNLO.QCD.pineappl.lz4",
        "dd2_NNLO.QCD.pineappl.lz4",
        "dd3_NNLO.QCD.pineappl.lz4"])

matrix_run.integrate_2d("ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR.pineappl.lz4", "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR_INTEGRATED.pineappl.lz4")



# TODO: how to integrate 2D dist?

# for i, n_bins in {1: 4, 2:4, 3:3}.items():
#     matrix_run.integrate_1d(
#         f"dd{i}_NNLO.QCD.pineappl.lz4",
#         temp_dir / f"dd{i}_NNLO.QCD_INTEGRATED.pineappl.lz4",
#         f"0-{n_bins-1}"
#     )
#
# run_pineappl_merge("ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR_INTEGRATED.pineappl.lz4", *list(temp_dir.glob("*.lz4")))
# matrix_run.integrate_1d(
#     "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR_INTEGRATED.pineappl.lz4",
#     "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR_INTEGRATED_test.pineappl.lz4",
#     "0-2"
# )

for exp in ["atlas", "cms"]:

    matrix_run = MatrixRun("run_ATLAS_TTBAR_8TEV_DIF", theory_id=theory_id)

    matrix_run.rename_grid(
        f"{exp}_2l_mttbar_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_2L_DIF_MTTBAR.pineappl.lz4",
    )
    matrix_run.integrate_1d(
        f"{exp}_2l_mttbar_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_2L_DIF_MTTBAR_INTEGRATED.pineappl.lz4",
    )

    matrix_run.rename_grid(
        f"{exp}_2l_yttbar_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_2L_DIF_YTTBAR.pineappl.lz4",
    )
    matrix_run.integrate_1d(
        f"{exp}_2l_yttbar_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_2L_DIF_YTTBAR_INTEGRATED.pineappl.lz4",
    )

    matrix_run.rename_grid(
        f"{exp}_lj_mttbar_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_MTTBAR.pineappl.lz4",
    )
    matrix_run.integrate_1d(
        f"{exp}_lj_mttbar_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_MTTBAR_INTEGRATED.pineappl.lz4",
    )

    matrix_run.rename_grid(
        f"{exp}_lj_pTt_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_PTT.pineappl.lz4",
    )
    matrix_run.integrate_1d(
        f"{exp}_lj_pTt_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_PTT_INTEGRATED.pineappl.lz4",
    )

    matrix_run.rename_grid(
        f"{exp}_lj_yt_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_YT.pineappl.lz4",
    )
    matrix_run.integrate_1d(
        f"{exp}_lj_yt_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_YT_INTEGRATED.pineappl.lz4",
    )

    matrix_run.rename_grid(
        f"{exp}_lj_yttbar_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_YTTBAR.pineappl.lz4",
    )
    matrix_run.integrate_1d(
        f"{exp}_lj_yttbar_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_YTTBAR_INTEGRATED.pineappl.lz4",
    )

    matrix_run.rename_grid(
        "total_rate_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_8TEV_TOT_X-SEC.pineappl.lz4",
    )

for exp in ["atlas", "cms"]:

    matrix_run = MatrixRun("run_ATLAS_TTBAR_7TEV_TOT_X-SEC", theory_id=theory_id)
    matrix_run.rename_grid(
        "total_rate_NNLO.QCD.pineappl.lz4",
        f"{exp.upper()}_TTBAR_7TEV_TOT_X-SEC.pineappl.lz4",
    )

matrix_run = MatrixRun("run_CMS_TTBAR_5TEV_TOT_X-SEC", theory_id=theory_id)
matrix_run.rename_grid(
    "total_rate_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_5TEV_TOT_X-SEC.pineappl.lz4"
)

matrix_run = MatrixRun("run_CMS_TTBAR_8TEV_2L_DIF", theory_id=theory_id)


# CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT
matrix_run.rename_grid(
    "mttbar-yt_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT.pineappl.lz4"
)

# CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT-INTEGRATED

# CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YTTBAR-INTEGRATED

# CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YTTBAR
matrix_run.rename_grid(
    "mttbar-yttbar_NNLO.QCD.pineappl.lz4",
    "CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YTTBAR.pineappl.lz4",
)

# CMS_TTBAR_8TEV_2L_DIF_PTT-YT-INTEGRATED

# CMS_TTBAR_8TEV_2L_DIF_PTT-YT
matrix_run.rename_grid(
    "pTt-yt_NNLO.QCD.pineappl.lz4", "CMS_TTBAR_8TEV_2L_DIF_PTT-YT .pineappl.lz4"
)
