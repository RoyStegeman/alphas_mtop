# this script updates the metadata of a PineAPPL grid file
# specifically, it modifies the "theory_card" entry to update the comments
import pathlib
import json
import pineappl
import subprocess
import requests
import getpass
import yaml
from tqdm import tqdm

password = getpass.getpass("Enter password: ")

path_to_fk_tables = pathlib.Path("/data/theorie/jthoeve/physics_projects/theories_slim/data/fktables")
path_to_theory_cards = pathlib.Path("/data/theorie/jthoeve/physics_projects/nnpdf/nnpdf_data/nnpdf_data/theory_cards")

#theories = list(range(40009004, 40009011))
theories = [40009000, 40009001, 40009002]
theories += list(range(40010000, 40010013))
theories += list(range(40011000, 40011013))
theories += list(range(40012000, 40012013))


for theory_id in theories:

    # download the theory from the CERNbox
    # print(f"Downloading theory {theory_id}...")
    #
    # url = f"https://cernbox.cern.ch/remote.php/dav/public-files/MsBioAS1FShShI9/theory_{theory_id}.tgz?signature=f59683a0c2f9d67e4c7efca76398e4345981ca949002cb05f2fe69c5c1350f25&expiration=2025-05-08T15%3A39%3A56%2B02%3A00"
    # output_file = f"theory_{theory_id}.tgz"
    #
    # response = requests.get(url, auth=('', password), stream=True)
    # response.raise_for_status()
    # total = int(response.headers.get('content-length', 0))
    # with open(output_file, 'wb') as f, tqdm(
    #         desc=output_file,
    #         total=total,
    #         unit='B',
    #         unit_scale=True,
    #         unit_divisor=1024,
    # ) as bar:
    #     for chunk in response.iter_content(chunk_size=8192):
    #         size = f.write(chunk)
    #         bar.update(size)
    #
    # # extract the tarball
    # print(f"Finished downloading theory {theory_id}...")
    # print("Extracting the tarball...")
    # subprocess.run(["tar", "-xf", output_file])
    # print("Finished extracting the tarball...")
    # # remove tarball
    # subprocess.run(["rm", output_file])

    path_to_fk_table = pathlib.Path("/data/theorie/jthoeve/physics_projects/theories_slim/cernbox/theory_{}".format(theory_id)) / "fastkernel"
    for fk_table in path_to_fk_table.iterdir():
        if "_TTBAR_" not in str(fk_table):
            continue
        if fk_table.suffix != ".lz4":
            continue

        grid = pineappl.grid.Grid.read(fk_table)
        metadata = grid.metadata

        theory_metadata = json.loads(metadata["theory_card"])

        with open(path_to_theory_cards / "{}.yaml".format(theory_id), "r") as f:
            theory_card = yaml.safe_load(f)
        new_comment = theory_card["Comments"]

        theory_metadata["Comments"] = new_comment
        theory_metadata = json.dumps(theory_metadata)
        grid.set_metadata("theory_card", theory_metadata)

        grid.write_lz4(str(fk_table))

    print(f"Updated metadata for {theory_id}")

    # make new tarball
    #print("Creating new tarball...")
    #subprocess.run(["tar", "-czf", f"theory_{theory_id}.tgz", f"theory_{theory_id}"])






