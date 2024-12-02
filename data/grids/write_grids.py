# -*- coding: utf-8 -*-
import pathlib
import subprocess

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

def copy_file(src: pathlib.Path, dest: pathlib.Path):
    """Copy a file from src to dest."""
    try:
        subprocess.run(["cp", src, dest], check=True)
        print(f"Copied: {src} -> {dest}")
    except subprocess.CalledProcessError as e:
        print(f"Error copying file: {e}")
        #raise

def get_nbins(src: pathlib.Path) -> int:
    """
    Get the number of bins in a PineAPPL grid file
    """


    result = subprocess.run(
        [
            "pineappl",
            "read",
            "--bins",
            src,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return  len(result.stdout.split("\n")) - 3



def run_pineappl_write(src: pathlib.Path, dest: pathlib.Path, merge_bins: str = None):
    """Run the 'pineappl write' command."""


    if merge_bins is None:
        try:
            n_bins = get_nbins(src)
            merge_bins = f"0-{n_bins-1}"
        except:
            print("Not able to extract number of bins")
            return

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
        #raise

def run_pineappl_merge(output: pathlib.Path, *inputs: pathlib.Path):
    """Run the 'pineappl merge' command."""
    try:
        subprocess.run(["pineappl", "merge", output, *inputs],check=True,)
        print(f"PineAPPL merge: {inputs} -> {output}")
    except subprocess.CalledProcessError as e:
        print(f"Error running pineappl merge: {e}")
        #raise

matrix_run = "run_ATLAS_TTBAR_13TEV_HADR_DIF"
input_dir = get_input_dir(matrix_run)

copy_file(input_dir / "m_ttx_NNLO.QCD.pineappl.lz4", output_dir / "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR.pineappl.lz4")

run_pineappl_write(
    input_dir / "m_ttx_NNLO.QCD.pineappl.lz4",
    output_dir / "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR_INTEGRATED.pineappl.lz4"
)

copy_file(input_dir / "y_ttx_NNLO.QCD.pineappl.lz4", output_dir / "ATLAS_TTBAR_13TEV_HADR_DIF_YTTBAR.pineappl.lz4")
run_pineappl_write(
    input_dir / "y_ttx_NNLO.QCD.pineappl.lz4",
    output_dir / "ATLAS_TTBAR_13TEV_HADR_DIF_YTTBAR_INTEGRATED.pineappl.lz4"
)

matrix_run = "run_ATLAS_TTBAR_13TEV_LJ_DIF"
input_dir = get_input_dir(matrix_run)

# ATLAS

# ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR
copy_file(input_dir / "atlas_mttbar_NNLO.QCD.pineappl.lz4", output_dir / "ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR.pineappl.lz4")

# ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-INTEGRATED
run_pineappl_write(
    input_dir / "atlas_mttbar_NNLO.QCD.pineappl.lz4",
    output_dir / "ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-INTEGRATED.pineappl.lz4"
)

# ATLAS_TTBAR_13TEV_LJ_DIF_PTT
copy_file(input_dir / "atlas_pTt_NNLO.QCD.pineappl.lz4", output_dir / "ATLAS_TTBAR_13TEV_LJ_DIF_PTT.pineappl.lz4")

# ATLAS_TTBAR_13TEV_LJ_DIF_PTT-INTEGRATED
run_pineappl_write(
    input_dir / "atlas_pTt_NNLO.QCD.pineappl.lz4",
    output_dir / "ATLAS_TTBAR_13TEV_LJ_DIF_PTT-INTEGRATED.pineappl.lz4"
)

# ATLAS_TTBAR_13TEV_LJ_DIF_YT
copy_file(input_dir / "atlas_yt_NNLO.QCD.pineappl.lz4", output_dir / "ATLAS_TTBAR_13TEV_LJ_DIF_YT.pineappl.lz4")

# ATLAS_TTBAR_13TEV_LJ_DIF_YT-INTEGRATED
run_pineappl_write(
    input_dir / "atlas_yt_NNLO.QCD.pineappl.lz4",
    output_dir / "ATLAS_TTBAR_13TEV_LJ_DIF_YT-INTEGRATED.pineappl.lz4"
)

# ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR
copy_file(input_dir / "atlas_yttbar_NNLO.QCD.pineappl.lz4", output_dir / "ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR.pineappl.lz4")

# ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR-INTEGRATED
run_pineappl_write(
    input_dir / "atlas_yttbar_NNLO.QCD.pineappl.lz4",
    output_dir / "ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR-INTEGRATED.pineappl.lz4"
)

# ATLAS_TTBAR_13TEV_TOT_X-SEC
copy_file(input_dir / "total_rate_NNLO.QCD.pineappl.lz4", output_dir / "ATLAS_TTBAR_13TEV_TOT_X-SEC.pineappl.lz4")

# CMS_TTBAR_13TEV_2L_DIF_MTTBAR
copy_file(input_dir / "cms_2l_mttbar_NNLO.QCD.pineappl.lz4", output_dir / "CMS_TTBAR_13TEV_2L_DIF_MTTBAR.pineappl.lz4")

# CMS_TTBAR_13TEV_2L_DIF_MTTBAR-INTEGRATED
run_pineappl_write(
    input_dir / "cms_2l_mttbar_NNLO.QCD.pineappl.lz4",
    output_dir / "CMS_TTBAR_13TEV_2L_DIF_MTTBAR-INTEGRATED.pineappl.lz4"
)

# CMS_TTBAR_13TEV_2L_DIF_PTT
copy_file(input_dir / "cms_2l_pTt_NNLO.QCD.pineappl.lz4", output_dir / "CMS_TTBAR_13TEV_2L_DIF_PTT.pineappl.lz4")

# CMS_TTBAR_13TEV_2L_DIF_PTT-INTEGRATED
run_pineappl_write(
    input_dir / "cms_2l_pTt_NNLO.QCD.pineappl.lz4",
    output_dir / "CMS_TTBAR_13TEV_2L_DIF_PTT-INTEGRATED.pineappl.lz4"
)

# CMS_TTBAR_13TEV_2L_DIF_YT
copy_file(input_dir / "cms_2l_yt_NNLO.QCD.pineappl.lz4", output_dir / "CMS_TTBAR_13TEV_2L_DIF_YT.pineappl.lz4")

# CMS_TTBAR_13TEV_2L_DIF_YT-INTEGRATED
run_pineappl_write(
    input_dir / "cms_2l_yt_NNLO.QCD.pineappl.lz4",
    output_dir / "CMS_TTBAR_13TEV_2L_DIF_YT-INTEGRATED.pineappl.lz4"
)

# CMS_TTBAR_13TEV_2L_DIF_YTTBAR
copy_file(input_dir / "cms_2l_yttbar_NNLO.QCD.pineappl.lz4", output_dir / "CMS_TTBAR_13TEV_2L_DIF_YTTBAR.pineappl.lz4")

# CMS_TTBAR_13TEV_2L_DIF_YTTBAR-INTEGRATED
run_pineappl_write(
    input_dir / "cms_2l_yttbar_NNLO.QCD.pineappl.lz4",
    output_dir / "CMS_TTBAR_13TEV_2L_DIF_YTTBAR-INTEGRATED.pineappl.lz4"
)

# CMS_TTBAR_13TEV_LJ_DIF_MTTBAR
copy_file(input_dir / "cms_lj_mttbar_NNLO.QCD.pineappl.lz4", output_dir / "CMS_TTBAR_13TEV_LJ_DIF_MTTBAR.pineappl.lz4")

# CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-INTEGRATED
run_pineappl_write(
    input_dir / "cms_lj_mttbar_NNLO.QCD.pineappl.lz4",
    output_dir / "CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-INTEGRATED.pineappl.lz4"
)

# CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-YTTBAR
copy_file(input_dir / "cms_lj_mttbar-yttbar_NNLO.QCD.pineappl.lz4", output_dir / "CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-YTTBAR.pineappl.lz4")

# CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-YTTBAR-INTEGRATED
# TODO: how to integrate 2D dist

# CMS_TTBAR_13TEV_LJ_DIF_PTT
copy_file(input_dir / "cms_lj_pTt_NNLO.QCD.pineappl.lz4", output_dir / "CMS_TTBAR_13TEV_LJ_DIF_PTT.pineappl.lz4")


# CMS_TTBAR_13TEV_LJ_DIF_PTT-INTEGRATED
run_pineappl_write(
    input_dir / "cms_lj_pTt_NNLO.QCD.pineappl.lz4",
    output_dir / "CMS_TTBAR_13TEV_LJ_DIF_PTT-INTEGRATED.pineappl.lz4"
)

# CMS_TTBAR_13TEV_LJ_DIF_YT
copy_file(input_dir / "cms_lj_yt_NNLO.QCD.pineappl.lz4", output_dir / "CMS_TTBAR_13TEV_LJ_DIF_YT.pineappl.lz4")

# CMS_TTBAR_13TEV_LJ_DIF_YT-INTEGRATED
run_pineappl_write(
    input_dir / "cms_lj_yt_NNLO.QCD.pineappl.lz4",
    output_dir / "CMS_TTBAR_13TEV_LJ_DIF_YT-INTEGRATED.pineappl.lz4"
)

# CMS_TTBAR_13TEV_LJ_DIF_YTTBAR
copy_file(input_dir / "cms_lj_yttbar_NNLO.QCD.pineappl.lz4", output_dir / "CMS_TTBAR_13TEV_LJ_DIF_YTTBAR.pineappl.lz4")

# CMS_TTBAR_13TEV_LJ_DIF_YTTBAR-INTEGRATED
run_pineappl_write(
    input_dir / "cms_lj_yttbar_NNLO.QCD.pineappl.lz4",
    output_dir / "CMS_TTBAR_13TEV_LJ_DIF_YTTBAR-INTEGRATED.pineappl.lz4"
)

# CMS_TTBAR_13TEV_TOT_X-SEC
copy_file(input_dir / "total_rate_NNLO.QCD.pineappl.lz4", output_dir / "CMS_TTBAR_13TEV_TOT_X-SEC.pineappl.lz4")


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

for exp in ["atlas", "cms"]:

    matrix_run = "run_ATLAS_TTBAR_8TEV_DIF"
    input_dir = get_input_dir(matrix_run)

    copy_file(input_dir / f"{exp}_2l_mttbar_NNLO.QCD.pineappl.lz4", output_dir / f"{exp.upper()}_TTBAR_8TEV_2L_DIF_MTTBAR.pineappl.lz4")
    run_pineappl_write(
        input_dir / f"{exp}_2l_mttbar_NNLO.QCD.pineappl.lz4",
        output_dir / f"{exp.upper()}_TTBAR_8TEV_2L_DIF_MTTBAR_INTEGRATED.pineappl.lz4"
    )

    copy_file(input_dir / f"{exp}_2l_yttbar_NNLO.QCD.pineappl.lz4", output_dir / f"{exp.upper()}_TTBAR_8TEV_2L_DIF_YTTBAR.pineappl.lz4")
    run_pineappl_write(
        input_dir / f"{exp}_2l_yttbar_NNLO.QCD.pineappl.lz4",
        output_dir / f"{exp.upper()}_TTBAR_8TEV_2L_DIF_YTTBAR_INTEGRATED.pineappl.lz4"
    )

    copy_file(input_dir / f"{exp}_lj_mttbar_NNLO.QCD.pineappl.lz4", output_dir / f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_MTTBAR.pineappl.lz4")
    run_pineappl_write(
        input_dir / f"{exp}_lj_mttbar_NNLO.QCD.pineappl.lz4",
        output_dir / f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_MTTBAR_INTEGRATED.pineappl.lz4"
    )

    copy_file(input_dir / f"{exp}_lj_pTt_NNLO.QCD.pineappl.lz4", output_dir / f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_PTT.pineappl.lz4")
    run_pineappl_write(
        input_dir / f"{exp}_lj_pTt_NNLO.QCD.pineappl.lz4",
        output_dir / f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_PTT_INTEGRATED.pineappl.lz4"
    )

    copy_file(input_dir / f"{exp}_lj_yt_NNLO.QCD.pineappl.lz4", output_dir / f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_YT.pineappl.lz4")
    run_pineappl_write(
        input_dir / f"{exp}_lj_yt_NNLO.QCD.pineappl.lz4",
        output_dir / f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_YT_INTEGRATED.pineappl.lz4"
    )

    copy_file(input_dir / f"{exp}_lj_yttbar_NNLO.QCD.pineappl.lz4", output_dir / f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_YTTBAR.pineappl.lz4")
    run_pineappl_write(
        input_dir / f"{exp}_lj_yttbar_NNLO.QCD.pineappl.lz4",
        output_dir / f"{exp.upper()}_TTBAR_8TEV_LJ_DIF_YTTBAR_INTEGRATED.pineappl.lz4"
    )

    copy_file(input_dir / "total_rate_NNLO.QCD.pineappl.lz4", output_dir / f"{exp.upper()}_TTBAR_8TEV_TOT_X-SEC.pineappl.lz4")

for exp in ["atlas", "cms"]:

    matrix_run = "run_ATLAS_TTBAR_7TEV_TOT_X-SEC"
    input_dir = get_input_dir(matrix_run)
    copy_file(input_dir / "total_rate_NNLO.QCD.pineappl.lz4", output_dir / f"{exp.upper()}_TTBAR_7TEV_TOT_X-SEC.pineappl.lz4")