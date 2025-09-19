# Author: J. J. ter Hoeve
# Date: June 2024
# Description: This script replaces all non TTBAR TRN FK tables in theories 40_009, 40_010, 40_011, and 40_012
#              with the corresponding EXA tables from theorie 40_008

import pathlib
import shutil

path_to_fk_tables_EXA = pathlib.Path("/project/theorie/jthoeve/miniconda3/envs/nnpdf_dev/share/NNPDF/theories/")
path_to_fk_tables_TRN = pathlib.Path("/data/theorie/jthoeve/physics_projects/theories_slim/cernbox")

mapping = {"000": "000", "001": "001", "002": "002", "003": "003", "004": "004", "006": "005", "007": "006",
           "008": "007", "009": "008", "010": "009", "011": "010", "012": "011", "013": "012"}

# for id_EXA, id_TRN in mapping.items():
#     source_path = path_to_fk_tables_EXA / f"theory_40008{id_EXA}" / "fastkernel"
#     cfactor_path =  path_to_fk_tables_EXA / f"theory_40008{id_EXA}" / "cfactor"

    # copy cfactor dir
    # for theory in [40010, 40011, 40012]:
    #     target_cfactor_path = path_to_fk_tables_TRN / f"theory_{theory}{id_TRN}" / "cfactor"
    #     print("Copy from", cfactor_path, "to", target_cfactor_path)
    #     shutil.copytree(cfactor_path, target_cfactor_path, dirs_exist_ok=True)

    # for fk_table_EXA in source_path.iterdir():
    #     # do not copy the TTBAR tables
    #     if "TTB" in fk_table_EXA.name:
    #         continue
    #     for theory in [40010, 40011, 40012]:
    #         target_path = path_to_fk_tables_TRN / f"theory_{theory}{id_TRN}" / "fastkernel"
    #         #print("Copy from", fk_table_EXA, "to", target_path)
    #         shutil.copy(fk_table_EXA, target_path)

# treat as = 0.118 separately
for id_TRN in ["000", "001", "002"]:
    source_path = path_to_fk_tables_EXA / f"theory_40008005" / "fastkernel"
    target_path = path_to_fk_tables_TRN / f"theory_40009{id_TRN}" / "fastkernel"
    for fk_table_EXA in source_path.iterdir():
        # do not copy the TTBAR tables
        if "TTB" in fk_table_EXA.name:
            continue
        print("Copy from", fk_table_EXA, "to", target_path)
        shutil.copy(fk_table_EXA, target_path)




            