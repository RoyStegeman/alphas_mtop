from pathlib import Path
import pandas as pd
import os

theory_id = 40006002
result_dir = Path("/Users/jaco/Documents/physics_projects/alphas_mtop/MATRIX/result_fixed_scale")
report_dir = Path(f"/Users/jaco/Documents/physics_projects/alphas_mtop/benchmarks/uncertainties_fixed_scale/{theory_id}")

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

for run_dir in result_dir.iterdir():
    dfs = {}
    if not run_dir.is_dir():
        continue

    table_dir = Path(f"{result_dir}/{run_dir.stem}/{theory_id}/NNLO-run/distributions__NNLO_QCD/")
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