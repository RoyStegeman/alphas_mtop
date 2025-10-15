from validphys.api import API
import pathlib
import sys

#dataset = sys.argv[1]
result_dir = pathlib.Path("/data/theorie/jthoeve/physics_projects/alphas_mtop/dataset_selection/results")
pdf = sys.argv[1]#f"250926-jth-dataset-selection-with-{dataset}"
dataset = pdf.split("with-")[1]
#t0pdfset = "250926-jth-dataset-selection-iter1"

dict_dataset_selection = dict(theoryid=40010006, pdf=pdf, use_cuts="internal", fits=[pdf],
                              use_t0=False, use_thcovmat_if_present=True, show_total=True )
print("Computing chi2 for dataset selection with pdf:", pdf)
print("Using dataset:", dataset)
chi2_df_table = API.fits_chi2_table(**dict_dataset_selection)
print("chi2 computed")
chi2_df_ttbar = chi2_df_table[chi2_df_table.index.get_level_values(1) == dataset]
chi2_ttbar = chi2_df_ttbar.iloc[:, 1].values[0]
chi2_df_tot = chi2_df_table.loc[("Total", "Total")]
chi2_tot = chi2_df_tot.iloc[1]
ndat = chi2_df_ttbar.iloc[:, 0].values[0]

# write to a file
with open(result_dir / pdf / "chi2.txt", "w") as f:
    f.write("ndat \t chi2 ttbar \t chi2 tot\n")
    f.write(f"{ndat}\t{chi2_ttbar}\t{chi2_tot}\n")