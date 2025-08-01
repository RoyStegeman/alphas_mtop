from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

result_dir = Path("./final_results")
report_dir = Path(f"./final_results/report")

def read_table(table):
    df = pd.read_table(table, sep="\s+", header=0, skiprows=0, index_col=False)
    df_columns = df.columns[1:]
    df = df.iloc[:, :-1]
    df.columns = df_columns
    df["rel MC error"] = df["central-error"] / df["scale-central"]
    # remove rel-down from df
    df = df.loc[:, ~df.columns.str.contains("rel-down")]
    df = df.loc[:, ~df.columns.str.contains("rel-up")]
    return df

dfs_all = {}
for run_dir in result_dir.iterdir():
    dfs = {}
    if not run_dir.is_dir():
        continue

    table_dir = Path(f"{result_dir}/{run_dir.stem}/NNLO-run/distributions__NNLO_QCD/")
    if not table_dir.exists():
        continue
    for table in table_dir.iterdir():
        if table.is_file():
            df = read_table(table)
            dfs[table.stem] = df

    if not report_dir.exists():
        report_dir.mkdir(parents=True)

    with open(report_dir / f"uncertainties_{run_dir.stem}.tex", "w") as f:
        f.write("\\documentclass[11pt]{article}")
        f.write("\\usepackage{booktabs}")
        f.write("\\usepackage{longtable}")
        f.write("\\usepackage[table]{xcolor}")
        f.write("\\usepackage[paperheight=20in, paperwidth=12in]{geometry}")
        f.write("\\begin{document}")
        f.write("\\renewcommand{\\arraystretch}{1.0}")

    cnt=0
    for df_name, df in dfs.items():

        with pd.option_context("max_colwidth", 1000):
            with open(report_dir / f"uncertainties_{run_dir.stem}.tex", "a", ) as f:
                f.write(df.to_latex(index=False, longtable=True,
                                           escape=False, float_format="%.2e", caption=df_name.replace("_", "\_")))
        cnt += 1



    with open(report_dir / f"uncertainties_{run_dir.stem}.tex", "a") as f:
        f.write("\\end{document}")

    # compile tex and keep only pdf
    os.system(f"pdflatex -output-directory={report_dir} {report_dir}/uncertainties_{run_dir.stem}.tex")
    os.system(f"rm {report_dir}/*.log {report_dir}/*.aux {report_dir}/*.tex")

    dfs_all[run_dir.stem] = dfs


def format_asymmetry(val):
    if isinstance(val, float) and abs(val) > 0.1:
        return f"\\cellcolor{{yellow!25}} {val:.2e}"
    return f"{val:.2e}"


asymmetries_good_mc = []
asymmetries_bad_mc = []
for sqrts in [8, 13]:
    dfs_plus = dfs_all[f"run_ttb_{sqrts}tev_mt_175"]
    dfs_central = dfs_all[f"run_ttb_{sqrts}tev_mt_172P5"]
    dfs_minus = dfs_all[f"run_ttb_{sqrts}tev_mt_170"]

    with open(report_dir / f"asymmetry_{sqrts}.tex", "w") as f:
        f.write("\\documentclass[11pt]{article}")
        f.write("\\usepackage{booktabs}")
        f.write("\\usepackage{longtable}")
        f.write("\\usepackage[table]{xcolor}")
        f.write("\\usepackage[landscape, paperheight=25in, paperwidth=20in]{geometry}")
        f.write("\\begin{document}")
        f.write("\\renewcommand{\\arraystretch}{1.0}")


    for run_name in dfs_plus.keys():
        if "ATLAS" not in run_name and "CMS" not in run_name:
            continue
        df_plus = dfs_plus[run_name]
        df_central = dfs_central[run_name]
        df_minus = dfs_minus[run_name]

        delta_plus = df_plus["scale-central"] - df_central["scale-central"]
        delta_minus = df_minus["scale-central"] - df_central["scale-central"]

        abs_error_plus = df_plus["central-error"]
        abs_error_min = df_minus["central-error"]

        asymmetry = (np.abs(delta_plus) - np.abs(delta_minus)) / (np.abs(delta_plus) + np.abs(delta_minus))

        dfs_central[run_name].rename(columns={"rel MC error": "rel MC error 0"}, inplace=True)
        dfs_central[run_name]["rel MC error -"] = df_minus["rel MC error"]
        dfs_central[run_name]["abs MC error -"] = abs_error_min
        dfs_central[run_name]["rel MC error +"] = df_plus["rel MC error"]
        dfs_central[run_name]["abs MC error +"] = abs_error_plus
        dfs_central[run_name]["rel delta +"] = delta_plus / df_central["scale-central"]
        dfs_central[run_name]["abs delta +"] = delta_plus
        dfs_central[run_name]["rel delta -"] = delta_minus / df_central["scale-central"]
        dfs_central[run_name]["abs delta -"] = delta_minus
        dfs_central[run_name]["rel error delta +"] = np.abs(abs_error_plus / delta_plus)
        dfs_central[run_name]["rel error delta -"] = np.abs(abs_error_min / delta_minus)

        dfs_central[run_name]["asymmetry"] = asymmetry
        formatters = {
            col: format_asymmetry if (col == "rel error delta -" or col == "rel error delta +")else lambda x: f"{x:.2e}" if isinstance(x, float) else x
            for col in dfs_central[run_name].columns
        }
        dfs_central[run_name].drop(columns=["scale-min", "scale-max", "min-error", "max-error", "central-error"], inplace=True)

        keep = (abs_error_plus/ delta_plus < 0.1) & (abs_error_min / delta_minus < 0.1)

        # Convert keep to boolean list aligned with DataFrame index
        highlight_rows = (~keep).reset_index(drop=True).tolist()


        asymmetries_good_mc.extend(asymmetry[keep].values)
        asymmetries_bad_mc.extend(asymmetry[~keep].values)


        # remove "scale-min" from the dataframe



        with pd.option_context("max_colwidth", 1000):
            with open(report_dir / f"asymmetry_{sqrts}.tex", "a") as f:
                f.write(dfs_central[run_name].to_latex(index=False, longtable=True,escape=False,
                                                       float_format="%.2e", caption=run_name.replace("_", "\_"),
                                                       formatters=formatters))


    with open(report_dir / f"asymmetry_{sqrts}.tex", "a") as f:
        f.write("\\end{document}")

    # compile tex and keep only pdf
    os.system(f"pdflatex -output-directory={report_dir} {report_dir}/asymmetry_{sqrts}.tex")
    os.system(f"rm {report_dir}/*.log {report_dir}/*.aux {report_dir}/*.tex")

fig, ax = plt.subplots(figsize=(10, 6))

asymmetries = np.array(asymmetries_good_mc)
asymmetries_bad_mc = np.array(asymmetries_bad_mc)
ax.hist([asymmetries_good_mc, asymmetries_bad_mc], alpha=0.5, label=["Good MC", "Bad MC"], bins=np.arange(-1, 1.1, 0.1), stacked=True)
plt.xlabel(r"$(\Delta T^+ - \Delta T^-)/(\Delta T^+ + \Delta T^-)$")
plt.ylabel(r"$\rm{Nr.\;of\;datapoints}$")
plt.legend()
plt.grid(True)
plt.savefig(report_dir / "asymmetry_histogram.pdf")