# -*- coding: utf-8 -*-
import pathlib
import subprocess

# Define directories
output_dir = pathlib.Path(".")
matrix_result_dir = pathlib.Path("../../MATRIX/result")
pineappl_dir = pathlib.Path("NNLO-run/PineAPPL_grids")

# Define Matrix theory details
theory = "40006000"
output_dir = output_dir / theory  # Ensure output directory is scoped to the theory
output_dir.mkdir(parents=True, exist_ok=True)
# temp_dir = output_dir / "temp"
# temp_dir.mkdir(parents=True, exist_ok=True)

def get_input_dir(matrix_run: str) -> pathlib.Path:
    """Construct the full input directory path."""
    return matrix_result_dir / matrix_run / theory / pineappl_dir

def copy_file(src: pathlib.Path, dest: pathlib.Path):
    """Copy a file from src to dest."""
    try:
        subprocess.run(["cp", src, dest], check=True)
        print(f"Copied: {src} -> {dest}")
    except subprocess.CalledProcessError as e:
        print(f"Error copying file: {e}")
        raise

def run_pineappl_write(src: pathlib.Path, dest: pathlib.Path, merge_bins: str):
    """Run the 'pineappl write' command."""
    try:
        subprocess.run(
            [
                "pineappl",
                "write",
                "--merge-bins",
                merge_bins,
                src,
                dest
            ],
            check=True,
        )
        print(f"PineAPPL write: {src} -> {dest} (merge bins: {merge_bins})")
    except subprocess.CalledProcessError as e:
        print(f"Error running pineappl write: {e}")
        raise

def run_pineappl_merge(output: pathlib.Path, *inputs: pathlib.Path):
    """Run the 'pineappl merge' command."""
    try:
        subprocess.run(
            ["pineappl", "merge", output, *inputs],
            check=True,
        )
        print(f"PineAPPL merge: {inputs} -> {output}")
    except subprocess.CalledProcessError as e:
        print(f"Error running pineappl merge: {e}")
        raise

# Process files for matrix run 1
matrix_run = "run_ATLAS_TTBAR_13TEV_HADR_DIF"
input_dir = get_input_dir(matrix_run)

copy_file(input_dir / "m_ttx_NNLO.QCD.pineappl.lz4", output_dir / "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR.pineappl.lz4")
run_pineappl_write(
    input_dir / "m_ttx_NNLO.QCD.pineappl.lz4",
    output_dir / "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR_INTEGRATED.pineappl.lz4",
    "0-7"
)

copy_file(input_dir / "y_ttx_NNLO.QCD.pineappl.lz4", output_dir / "ATLAS_TTBAR_13TEV_HADR_DIF_YTTBAR.pineappl.lz4")
run_pineappl_write(
    input_dir / "y_ttx_NNLO.QCD.pineappl.lz4",
    output_dir / "ATLAS_TTBAR_13TEV_HADR_DIF_YTTBAR_INTEGRATED.pineappl.lz4",
    "0-11"
)

# Process files for matrix run 2
matrix_run = "run_ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR"
input_dir = get_input_dir(matrix_run)

run_pineappl_merge(
    output_dir / "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR.pineappl.lz4",
    input_dir / "dd1_NNLO.QCD.pineappl.lz4",
    input_dir / "dd2_NNLO.QCD.pineappl.lz4",
    input_dir / "dd3_NNLO.QCD.pineappl.lz4"
)

# TODO: how to integrate 2D dist?

# for i, n_bins in {1: 4, 2:4, 3:3}.items():
#     run_pineappl_write(
#         input_dir / f"dd{i}_NNLO.QCD.pineappl.lz4",
#         temp_dir / f"dd{i}_NNLO.QCD_INTEGRATED.pineappl.lz4",
#         f"0-{n_bins-1}"
#     )
#
# run_pineappl_merge(output_dir / "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR_INTEGRATED.pineappl.lz4", *list(temp_dir.glob("*.lz4")))
# run_pineappl_write(
#     output_dir / "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR_INTEGRATED.pineappl.lz4",
#     output_dir / "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR_INTEGRATED_test.pineappl.lz4",
#     "0-2"
# )

matrix_run = "run_ATLAS_TTBAR_8TEV_DIF"
input_dir = get_input_dir(matrix_run)

copy_file(input_dir / "atlas_2l_mttbar_NNLO.QCD.pineappl.lz4", output_dir / "ATLAS_TTBAR_8TEV_2L_DIF_MTTBAR.pineappl.lz4")
run_pineappl_write(
    input_dir / "atlas_2l_mttbar_NNLO.QCD.pineappl.lz4",
    output_dir / "ATLAS_TTBAR_8TEV_2L_DIF_MTTBAR_INTEGRATED.pineappl.lz4",
    "0-5"
)

copy_file(input_dir / "atlas_2l_yttbar_NNLO.QCD.pineappl.lz4", output_dir / "ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR.pineappl.lz4")
run_pineappl_write(
    input_dir / "atlas_2l_yttbar_NNLO.QCD.pineappl.lz4",
    output_dir / "ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR_INTEGRATED.pineappl.lz4",
    "0-4"
)

copy_file(input_dir / "atlas_lj_mttbar_NNLO.QCD.pineappl.lz4", output_dir / "ATLAS_TTBAR_8TEV_LJ_DIF_MTTBAR.pineappl.lz4")
run_pineappl_write(
    input_dir / "atlas_lj_mttbar_NNLO.QCD.pineappl.lz4",
    output_dir / "ATLAS_TTBAR_8TEV_LJ_DIF_MTTBAR_INTEGRATED.pineappl.lz4",
    "0-6"
)

copy_file(input_dir / "atlas_lj_pTt_NNLO.QCD.pineappl.lz4", output_dir / "ATLAS_TTBAR_8TEV_LJ_DIF_PTT.pineappl.lz4")
run_pineappl_write(
    input_dir / "atlas_lj_pTt_NNLO.QCD.pineappl.lz4",
    output_dir / "ATLAS_TTBAR_8TEV_LJ_DIF_PTT_INTEGRATED.pineappl.lz4",
    "0-7"
)

copy_file(input_dir / "atlas_lj_yt_NNLO.QCD.pineappl.lz4", output_dir / "ATLAS_TTBAR_8TEV_LJ_DIF_YT.pineappl.lz4")
run_pineappl_write(
    input_dir / "atlas_lj_yt_NNLO.QCD.pineappl.lz4",
    output_dir / "ATLAS_TTBAR_8TEV_LJ_DIF_YT_INTEGRATED.pineappl.lz4",
    "0-4"
)

copy_file(input_dir / "atlas_lj_yttbar_NNLO.QCD.pineappl.lz4", output_dir / "ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR.pineappl.lz4")
run_pineappl_write(
    input_dir / "atlas_lj_yttbar_NNLO.QCD.pineappl.lz4",
    output_dir / "ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR_INTEGRATED.pineappl.lz4",
    "0-4"
)