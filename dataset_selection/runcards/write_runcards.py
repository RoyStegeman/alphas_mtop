import ruyaml
from pathlib import Path

yaml = ruyaml.YAML()

# Load YAML from a file
input_file = "250926-jth-dataset-selection.yaml"
with open(input_file, "r") as f:
    mock_runcard = yaml.load(f)

# Use the correct key from the YAML file
dataset_key = 'dataset_inputs'
all_datasets = mock_runcard[dataset_key]

# Separate TTBAR and non-TTBAR datasets
non_ttbar = [d for d in all_datasets if "TTBAR" not in d["dataset"]]
ttbar_datasets = [d for d in all_datasets if "TTBAR" in d["dataset"]]

# Output directory (same as input file)
out_dir = Path(input_file).parent

for ttbar in ttbar_datasets:
    new_runcard = dict(mock_runcard)  # shallow copy is fine since we replace the list
    new_runcard[dataset_key] = non_ttbar + [ttbar]
    ttbar_name = ttbar["dataset"]
    # Update the description to mention the added TTBAR dataset
    orig_desc = mock_runcard.get("description", "")
    new_runcard["description"] = f"{orig_desc} [TTBAR dataset added: {ttbar_name}]"
    # Clean up the name for filename
    ttbar_name_safe = ttbar_name.replace('/', '_').replace(':', '_')
    out_file = out_dir / f"250926-jth-dataset-selection-with-{ttbar_name_safe}.yaml"
    with open(out_file, "w") as f:
        yaml.dump(new_runcard, f)
    print(f"Wrote {out_file}")
