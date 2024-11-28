import pathlib
import subprocess

# Define directories
output_dir = pathlib.Path("../data/grids")
matrix_result_dir = pathlib.Path("../MATRIX/result")
pineappl_dir = pathlib.Path("NNLO-RUN/PineAPPL_grids")

# Define Matrix run and theory details
matrix_run = "run_ATLAS_TTBAR_13TEV_HADR_DIF"
theory = "40006000"

# Mapping of input to output files
mapping = {
    "m_ttx_NNLO.QCD.pineappl.lz4": {"name": "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR.pineappl.lz4", "n_bins": 8},
    "y_ttx_NNLO.QCD.pineappl.lz4": {"name": "ATLAS_TTBAR_13TEV_HADR_DIF_YTTBAR.pineappl.lz4", "n_bins": 12}
}

# Construct the full source path
full_path = matrix_result_dir / matrix_run / theory / pineappl_dir

# Ensure the output directory exists
output_dir = output_dir / theory
output_dir.mkdir(parents=True, exist_ok=True)


# Copy files as per the mapping
for input_name, output_dict in mapping.items():
    src_file = full_path / input_name
    dest_file = output_dir / output_dict["name"]
    n_bins = output_dict["n_bins"]

    try:
        subprocess.run(["cp", str(src_file), str(dest_file)], check=True)
        print(f"Produced {dest_file}")
        subprocess.run(["pineappl", "write", "--merge-bins", f"0-{n_bins-1}", str(src_file), str(dest_file).replace(".pineappl.lz4", "_INTEGRATED.pineappl.lz4")], check=True)
        print(f"Integrated {dest_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error copying {src_file} to {dest_file}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


# cp $FULL_PATH/"m_ttx_NNLO.QCD.pineappl.lz4" /Users/jaco/Documents/physics_projects/alphas_mtop/data/grids/40006000/ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR.pineappl.lz4
#
# pineappl write --set-key-file dddistribution.dat /Users/jaco/Documents/physics_projects/alphas_mtop/MATRIX/result/run_ATLAS_TTBAR_13TEV_HADR_DIF/000/input_of_run/dddistribution.dat /Users/jaco/Documents/physics_projects/alphas_mtop/data/grids/40006000/ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR.pineappl.lz4 /Users/jaco/Documents/physics_projects/alphas_mtop/data/grids/40006000/ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR_test.pineappl.lz4
#
