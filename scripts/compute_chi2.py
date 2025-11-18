from validphys.api import API
import pathlib
import sys
import pandas as pd

result_dir = pathlib.Path("/data/theorie/jthoeve/physics_projects/alphas_mtop/results/tcm_results")
pdf = sys.argv[1]

theoryid = 40013005 if "N3LO" in pdf else 40010006

dict_dataset_selection = dict(theoryid=theoryid, pdf=pdf, use_cuts="internal", fits=[pdf],
                              use_t0=False, use_thcovmat_if_present=True, show_total=True, 
                              metadata_group="nnpdf31_process")

print("Computing chi2 for dataset selection with pdf:", pdf)


chi2_df_table_group = API.fits_groups_chi2_table(**dict_dataset_selection)
chi2_df_table_dataset = API.fits_chi2_table(**dict_dataset_selection)

# filter out TTBAR datasets
ttbar_datasets = ["TTBAR" in dataset for dataset in chi2_df_table_dataset.index.get_level_values(1)]
chi2_df_table = chi2_df_table_dataset[ttbar_datasets].droplevel(0)
chi2_df_table.columns = chi2_df_table.columns.droplevel(0)
chi2_df_table_TTBAR = chi2_df_table_group.loc["TOP"].droplevel(0)

z_scores = {}
for dataset in chi2_df_table.index:
    z = API.covmat_stability_characteristic(dataset_input={'dataset': dataset}, theoryid=theoryid,use_cuts="internal")
    z_scores[dataset] = z

z_scores = pd.Series(z_scores)

chi2_df_table = pd.concat([chi2_df_table.T, chi2_df_table_TTBAR], axis=1).T
chi2_df_table["z-score"] = z_scores

chi2_df_table.to_csv(result_dir / pdf / "chi2_table.csv", sep="\t")


