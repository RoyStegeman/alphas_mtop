# this script adds the MC uncertainty as a variant to the commondata

"""
variants:
    mc_uncertainties:
        data_uncertainties:
            - uncertainties.yaml
            - mc_uncertainties.yaml
"""

import pandas as pd
import pathlib
import numpy as np
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True
yaml.width = 4096  # set a very large width to prevent wrapping

# top datasets
dataset_inputs = {
"ATLAS_TTBAR_7TEV_TOT_X-SEC",
"ATLAS_TTBAR_8TEV_2L_DIF_MTTBAR-NORM",
"ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR-NORM",
"ATLAS_TTBAR_8TEV_LJ_DIF_MTTBAR-NORM",
"ATLAS_TTBAR_8TEV_LJ_DIF_PTT-NORM",
"ATLAS_TTBAR_8TEV_LJ_DIF_YT-NORM",
"ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM",
"ATLAS_TTBAR_8TEV_TOT_X-SEC",
"ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-NORM",
"ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR-NORM",
"ATLAS_TTBAR_13TEV_HADR_DIF_YTTBAR-NORM",
"ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-NORM",
"ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-PTT-NORM",
"ATLAS_TTBAR_13TEV_LJ_DIF_PTT-NORM",
"ATLAS_TTBAR_13TEV_LJ_DIF_YT-NORM",
"ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR-NORM",
"CMS_TTBAR_5TEV_TOT_X-SEC",
"CMS_TTBAR_7TEV_TOT_X-SEC",
"CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT-NORM",
"CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YTTBAR-NORM",
"CMS_TTBAR_8TEV_2L_DIF_PTT-YT-NORM",
"CMS_TTBAR_8TEV_LJ_DIF_MTTBAR-NORM",
"CMS_TTBAR_8TEV_LJ_DIF_PTT-NORM",
"CMS_TTBAR_8TEV_LJ_DIF_YT-NORM",
"CMS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM",
"CMS_TTBAR_8TEV_TOT_X-SEC",
"CMS_TTBAR_13TEV_2L_DIF_MTTBAR-NORM",
"CMS_TTBAR_13TEV_2L_DIF_PTT-NORM",
"CMS_TTBAR_13TEV_2L_DIF_YT-NORM",
"CMS_TTBAR_13TEV_2L_DIF_YTTBAR-NORM",
"CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-NORM",
"CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-YTTBAR-NORM",
"CMS_TTBAR_13TEV_LJ_DIF_PTT-NORM",
"CMS_TTBAR_13TEV_LJ_DIF_YT-NORM",
"CMS_TTBAR_13TEV_LJ_DIF_YTTBAR-NORM",
"CMS_TTBAR_13TEV_TOT_X-SEC"
}

# the following datasets were computed only for yttbar > 0, but the exp data also contains yttbar <0
# so treat these as a special case
yttbar_reflected = ["CMS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM", "CMS_TTBAR_13TEV_2L_DIF_YTTBAR-NORM"]

matrix_suffix = "__NNLO_QCD"
matrix_prefix_2D = "2D_"
fb_to_pb = 1e-3
matrix_run_dir = pathlib.Path("../MATRIX/final_results")
commondata_dir = pathlib.Path("/Users/jaco/Documents/physics_projects/nnpdf/nnpdf_data/nnpdf_data/commondata")

def read_table(table):
    """
     Reads Monte Carlo uncertainty from MATRIX and returns a dictionary of dataframes
    """
    df = pd.read_table(table, sep="\s+", header=0, skiprows=0, index_col=False)
    df_columns = df.columns[1:]
    df = df.iloc[:, :-1]
    df.columns = df_columns
    return df

def convert_unc_abs_to_norm(df):
    """
    propagates the error on x_i to y_i = x_i / S with S = sum(x_i)
    """
    x_i = df["scale-central"]
    delta_x_i = df["central-error"]
    S = x_i.sum(axis=0)
    sum_delta_xi_sq = (delta_x_i ** 2).sum(axis=0)

    # propagate error
    delta_y_i_sq = (delta_x_i ** 2 - 2 * x_i * delta_x_i ** 2 / S + x_i ** 2 / S ** 2 * sum_delta_xi_sq) / S ** 2
    delta_y_i = np.sqrt(delta_y_i_sq)
    y_i = x_i / S
    return y_i, delta_y_i

def write_commondata_unc(mc_uncs, commondata_path, observable_name):
    """
    Writes the Monte Carlo uncertainties to uncertainties.yaml in commondata format
    """
    mc_dict = {"definitions": {"mc_stat":
                                   {"description": "Monte Carlo uncertainty on the mtop variations",
                                    "treatment": "ADD",
                                    "type": "UNCORR"}}, "bins": [{"mc_stat": mc_unc} for mc_unc in mc_uncs]}

    # link to metadata
    with open(commondata_path / "metadata.yaml") as file:
        metadata = yaml.load(file)

    for obs in metadata["implemented_observables"]:
        if obs["observable_name"] == observable_name:
            data_uncertainties = obs["data_uncertainties"]

            # associate the observable name to the mc uncertainty file
            unc_suffix = data_uncertainties[0].split('uncertainties_')[1]

            # check if the number of bins is the same
            with open(commondata_path / f"kinematics_{unc_suffix}") as file:
                kinematics = yaml.load(file)
            if len(kinematics["bins"]) != len(mc_dict["bins"]):
                print(f"Number of bins in kinematics ({len(kinematics['bins'])}) does not match "
                                 f"the number of bins in mc uncertainties ({len(mc_dict['bins'])}) for "
                                 f"{commondata_path.stem + "_" + observable_name}")
                print("Removing the last bin in the mc uncertainties...")

                mc_dict["bins"] = mc_dict["bins"][:-1]

            mc_unc_filename = f"mc_top_uncertainties_{unc_suffix}"
            with open(commondata_path / mc_unc_filename, 'w') as file:
                yaml.dump(mc_dict, file)

            data_uncertainties_variant = data_uncertainties + [mc_unc_filename]
            if "variants" in obs:
                obs["variants"]["mc_uncertainties"] = {"data_uncertainties": data_uncertainties_variant}
            else:
                obs["variants"] = {"mc_uncertainties": {"data_uncertainties": data_uncertainties_variant}}
        else:
            continue

    # dump updated metadata
    with open(commondata_path / "metadata.yaml", 'w') as file:
        yaml.dump(metadata, file)



dfs_all = {}
for run_dir in matrix_run_dir.iterdir():
    dfs = {}
    if not run_dir.is_dir():
        continue

    table_dir = pathlib.Path(f"{matrix_run_dir}/{run_dir.stem}/NNLO-run/distributions__NNLO_QCD/")
    if not table_dir.exists():
        continue
    for table in table_dir.iterdir():
        if table.is_file():
            df = read_table(table)
            dfs[table.stem] = df

    dfs_all[run_dir.stem] = dfs

for dataset in dataset_inputs:

    commondata_name, observable_name = dataset.rsplit("_", 1)
    commondata_path = commondata_dir / commondata_name

    # Total xsec are computed with top++, not matrix
    if "TOT_X-SEC" in dataset:
        continue

    sqrts = int(dataset.split("_")[2].replace("TEV", ""))
    mt_vals = [170, 175]

    # collect MC for each mt variation
    delta_y_i_mt = {}
    for mt_val in mt_vals:

        if "NORM" in dataset:

            # find the uncertainty in each bin and the sum over bins
            dataset_abs = dataset.split("-NORM")[0]
            if dataset_abs.count("-") > 0:  # 2D dist
                matrix_filename = matrix_prefix_2D + dataset_abs + matrix_suffix
            elif "YTTBAR" in dataset:  # dist in yttbar
                matrix_filename = dataset_abs + "_catr" + matrix_suffix
            else:
                matrix_filename = dataset_abs + matrix_suffix

            # the runs ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR-NORM were split in 3, concatenate first
            if dataset == "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR-NORM":
                dfs = []
                for i in [1, 2, 3]:
                    matrix_filename_gm = matrix_filename.replace("__NNLO_QCD", f"_gm{i}__NNLO_QCD")
                    df_gm = dfs_all[f'run_ttb_{sqrts}tev_mt_{mt_val}'][matrix_filename_gm]
                    dfs.append(df_gm)
                df = pd.concat(dfs, ignore_index=True)
            # the bins in ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-PTT-NORM need to be merged first
            elif dataset == "ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-PTT-NORM":
                matrix_filename_bm = matrix_filename.replace("__NNLO_QCD", f"_bm__NNLO_QCD")
                df_bm = dfs_all[f'run_ttb_{sqrts}tev_mt_{mt_val}'][matrix_filename_bm]
                df_bm = df_bm[["scale-central", "central-error"]]
                merged_bins = [(0, 1), (2, 3), (4, 7), (9, 10), (11, 13), (14, 15), (17, 18), (19, 20), (21, 22), (24, 27), (28, 29), (30, 31)]
                scale_central_merged = [df_bm.loc[start:end, "scale-central"].sum() for start, end in merged_bins]
                central_error_merged = [np.sqrt((df_bm.loc[start:end, "central-error"] ** 2).sum()) for start, end in merged_bins]
                df = pd.DataFrame({"scale-central": scale_central_merged, "central-error": central_error_merged})
            elif matrix_filename in dfs_all[f'run_ttb_{sqrts}tev_mt_{mt_val}'].keys():
                df = dfs_all[f'run_ttb_{sqrts}tev_mt_{mt_val}'][matrix_filename]

            _, delta_y_i = convert_unc_abs_to_norm(df)
            delta_y_i_mt[mt_val] = delta_y_i

    # average mc unc in top variations
    try:
        delta_y_i_avg = np.sqrt(0.5 * (delta_y_i_mt[170] ** 2 + delta_y_i_mt[175] ** 2))
    except KeyError:
        # TODO: 2D grids with bm and gm as suffix are not recognized
        continue

    # 1D dist in yttbar
    if dataset in yttbar_reflected:
        delta_y_i_avg = pd.concat([delta_y_i_avg.iloc[::-1], delta_y_i_avg], ignore_index=True)

    # write to commondata folder

    write_commondata_unc(delta_y_i_avg, commondata_path, observable_name)



