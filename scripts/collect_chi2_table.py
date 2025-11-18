import pathlib
import pandas as pd

result_dir = pathlib.Path("/data/theorie/jthoeve/physics_projects/alphas_mtop/results/tcm_results")

results = {}
obs_set = set()
for result in result_dir.iterdir():
    if "251118" in result.stem:
        try:
            chi2_table = pd.read_csv(result / "chi2_table.csv", sep="\t", index_col=0)
            chi2_table.index = [idx.split("_DIF")[0] for idx in chi2_table.index]
        
            obs = result.stem.split("TTBAR_")[-1]
            obs_set.add(obs)
            obs_index = [obs] * len(chi2_table)
            chi2_table.index = pd.MultiIndex.from_arrays([obs_index, chi2_table.index],
                                                 names=["observable", "dataset"])
            results[result.stem] = chi2_table
      
        except FileNotFoundError:
            print(f"File not found: {result / 'chi2_table.csv'}")

combined_results = pd.concat(results.values())

df_all = pd.DataFrame()

for obs in sorted(obs_set):

    keys = {
        "ATLAS":      f'251118-jth-dataset-selection-N3LO-with-ATLAS_TTBAR_{obs}',
        "CMS":        f'251118-jth-dataset-selection-N3LO-with-CMS_TTBAR_{obs}',
        "ATLAS + CMS":  f'251118-jth-dataset-selection-N3LO-with-ATLAS_CMS_TTBAR_{obs}',
    }

    # --- Step 1: collect existing indices ---------------------------
    index_union = None
    for key in keys.values():
        df = results.get(key)
        if df is not None:
            idx = df.index
            index_union = idx if index_union is None else index_union.union(idx)

    if index_union is None:
        print(f"No data at all for observable: {obs}")
        continue

    # --- Step 2: build 3 columns, introducing NaNs if missing -------
    series_list = []
    for col_name, key in keys.items():
        df = results.get(key)
        if df is None:
            # create a NaN series with the full index
            s = pd.Series([float("nan")] * len(index_union),
                          index=index_union, name=col_name)
        else:
            # take chi2 column, reindex to full index (missing rows become NaN)
            s = df["$\\chi^2/ndata$"].reindex(index_union).rename(col_name)
        
        series_list.append(s)

    # --- Step 3: combine into one dataframe for this observable ----
    df_combined = pd.concat(series_list, axis=1)

    # add to global dataframe
    df_all = pd.concat([df_all, df_combined])

# tidy output
df_all = df_all.sort_index()

latex_names_dataset = {
    "ATLAS_TTBAR_13TEV_HADR": r"ATLAS 13~TeV $t\bar{t}$ all hadr.",
    "ATLAS_TTBAR_13TEV_LJ":   r"ATLAS 13~TeV $t\bar{t}$ $\ell+j$",
    "ATLAS_TTBAR_8TEV_2L":    r"ATLAS 8~TeV $t\bar{t}$ $2\ell$",
    "ATLAS_TTBAR_8TEV_LJ":    r"ATLAS 8~TeV $t\bar{t}$ $\ell+j$ ",
    "CMS_TTBAR_13TEV_2L_138FB-1": r"CMS 13~TeV $t\bar{t}$ $2\ell$ 138 ${\rm fb}^{-1}$",
    "CMS_TTBAR_13TEV_LJ":     r"CMS 13~TeV $t\bar{t}$ $\ell+j$",
    "CMS_TTBAR_13TEV_2L":   r"CMS 13~TeV $t\bar{t}$ $2\ell$",
    "TOP": r"total",
}

latex_names_observables = {
    "MTTBAR": r"$m_{t\bar{t}}$",
    "PTT":    r"$p_T^t$",
    "YTTBAR":  r"$y_{t\bar{t}}$",
    "YT":     r"$y_t$",
    "MTTBAR-PTT": r"$(m_{t\bar{t}}, p_T^t)$",
    "MTTBAR-YTTBAR": r"$(m_{t\bar{t}}, y_{t\bar{t}})$",
}

df_all = df_all.rename(index=latex_names_dataset, level="dataset")
df_all = df_all.rename(index=latex_names_observables, level="observable")
df_latex = df_all.fillna("—")
df_latex.to_latex("/data/theorie/jthoeve/physics_projects/alphas_mtop/results/tables/chi2_table.tex",
                  multirow=True,
                  multicolumn=True,
                  escape=False,
                  float_format="%.3f"
                  )
