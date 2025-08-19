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

pt_test = np.array([0.002639,
0.005891,
0.005954,
0.004422,
0.002644,
0.001431,
6.64500000e-04,
3.16700000e-04,
1.66100000e-04,
7.39400000e-05,
4.75e-05,
1.85e-05,
6.65100000e-06,
2.15900000e-06,
6.50600000e-07,
7.87200000e-08,
])

bin_width = np.array([40, 40, 40, 40, 40, 50, 50, 50, 50, 50, 50, 100, 100, 150, 150, 500])

mtt_test_exp = np.array([1.98085000e-03, 3.99172000e-03, 2.00949000e-03, 6.36295000e-04, 1.43808000e-04, 2.71952000e-05])
mtt_test_theory = np.array([1.2951925e0, 3.3421510e0, 1.7010997e0,5.4686545e-1,1.3677893e-1,2.6385245e-2,2.4443162e-3])
mtt_integrated = 8.2024355e2
mtt_test_theory_norm = mtt_test_theory / mtt_integrated
ratio = mtt_test_exp / mtt_test_theory_norm[:-1]
# I need to do mtt_test_theory/m

bin_width_mtt = np.array([80, 90, 150, 200, 280, 400])

# finding: np.sum(pt_test * bin_width) = 1


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
    propagates the error
    """
    x_i = df["scale-central"] # \Delta sigma_i / bin_width
    err_x_i = df["central-error"] # err (\Delta sigma_i / bin_width)
    if "left-edge" in df.columns and "right-edge" in df.columns:
        bin_width = df["right-edge"] - df["left-edge"]
    else:
        try:
            bin_width_1 = df["right-edge1"] - df["left-edge1"]
            bin_width_2 = df["right-edge2"] - df["left-edge2"]
            bin_width = bin_width_1 * bin_width_2
        except KeyError:
            import pdb; pdb.set_trace()

    delta_sigma_i = x_i * bin_width
    err_delta_sigma_i = err_x_i * bin_width

    S = delta_sigma_i.sum(axis=0)
    sum_err_delta_sigma_i_sq = (err_delta_sigma_i ** 2).sum(axis=0)

    # propagate error
    err_sq_delta_sigma_i_norm = (err_delta_sigma_i ** 2 - 2 * delta_sigma_i * err_delta_sigma_i ** 2 / S + delta_sigma_i ** 2 / S ** 2 * sum_err_delta_sigma_i_sq) / S ** 2
    err_delta_sigma_i_norm = np.sqrt(err_sq_delta_sigma_i_norm)

    return delta_sigma_i / S / bin_width, err_delta_sigma_i_norm / bin_width

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
    #mt_vals = ["172P5"]

    # collect MC for each mt variation
    delta_y_i_mt = {}
    for mt_val in mt_vals:

        if "NORM" in dataset:

            # find the uncertainty in each bin and the sum over bins
            dataset_abs = dataset.split("-NORM")[0]
            if dataset_abs.count("-") > 0:  # 2D dist
                matrix_filename = matrix_prefix_2D + dataset_abs + matrix_suffix
            elif dataset in yttbar_reflected:  # dist in yttbar
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
                df_bm = df_bm[["scale-central", "central-error", "left-edge1", "right-edge1",
                               "left-edge2", "right-edge2"]]
                merged_bins = [(0, 1), (2, 3), (4, 7), (8, 8), (9, 10), (11, 13), (14, 15), (16, 16), (17, 18),
                               (19, 20), (21, 22), (23, 23), (24, 27), (28, 29), (30, 31)]

                scale_central_merged = [df_bm.loc[start:end, "scale-central"].sum() for start, end in merged_bins]
                central_error_merged = [np.sqrt((df_bm.loc[start:end, "central-error"] ** 2).sum()) for start, end in merged_bins]
                bin_merged_1 = [(df_bm.loc[start, "left-edge1"], df_bm.loc[end, "right-edge1"]) for start, end in merged_bins]
                bin_merged_2 = [(df_bm.loc[start, "left-edge2"], df_bm.loc[end, "right-edge2"]) for start, end in merged_bins]
                left_edge1_merged = [bin_merged_1[i][0] for i in range(len(bin_merged_1))]
                right_edge1_merged = [bin_merged_1[i][1] for i in range(len(bin_merged_1))]
                left_edge2_merged = [bin_merged_2[i][0] for i in range(len(bin_merged_2))]
                right_edge2_merged = [bin_merged_2[i][1] for i in range(len(bin_merged_2))]


                df = pd.DataFrame({"scale-central": scale_central_merged, "central-error": central_error_merged,
                                   "left-edge1": left_edge1_merged, "right-edge1": right_edge1_merged,
                                   "left-edge2": left_edge2_merged, "right-edge2": right_edge2_merged})
            elif matrix_filename in dfs_all[f'run_ttb_{sqrts}tev_mt_{mt_val}'].keys():
                df = dfs_all[f'run_ttb_{sqrts}tev_mt_{mt_val}'][matrix_filename]


            y_i, delta_y_i = convert_unc_abs_to_norm(df)

            delta_y_i_mt[mt_val] = delta_y_i

    # average mc unc in top variations
    try:
        delta_y_i_avg = np.sqrt(0.5 * (delta_y_i_mt[170] ** 2 + delta_y_i_mt[175] ** 2))
    except KeyError:
        print(f"Skipping dataset {dataset}")
        continue
    #delta_y_i_avg = delta_y_i_mt["172P5"]



    # 1D dist in yttbar
    if dataset in yttbar_reflected:
        delta_y_i_avg = pd.concat([delta_y_i_avg.iloc[::-1], delta_y_i_avg], ignore_index=True)



    # write to commondata folder

    write_commondata_unc(delta_y_i_avg, commondata_path, observable_name)



