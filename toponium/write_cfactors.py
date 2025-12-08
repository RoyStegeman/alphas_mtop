import pathlib
from validphys.api import API
import pineappl
import numpy as np
import lhapdf
from pineappl.convolutions import Conv, ConvType

result_dir = pathlib.Path("./c_factors")
conv_type = ConvType(polarized=False, time_like=False)
conv_object = Conv(convolution_types=conv_type, pid=2212)
pdfset = "NNPDF40_nnlo_as_01180"
# Use the central PDF, ie. member ID=0
pdf = lhapdf.mkPDF(pdfset, 0)

fk_table_dir = pathlib.Path("/Users/jaco/miniconda3/envs/nnpdf_dev/share/NNPDF/theories")
theoryid = 40009000
dataset_inputs = [{"dataset": "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR"},
                  {"dataset": "ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR"},
                  {"dataset": "ATLAS_TTBAR_8TEV_2L_DIF_MTTBAR"},
                  {"dataset": "ATLAS_TTBAR_8TEV_LJ_DIF_MTTBAR"},
                  {"dataset": "CMS_TTBAR_13TEV_2L_138FB-1_DIF_MTTBAR"},
                  {"dataset": "CMS_TTBAR_13TEV_LJ_DIF_MTTBAR"}]
                  # {"dataset": "CMS_TTBAR_13TEV_2L_138FB-1_DIF_MTTBAR-YTTBAR"},
                  # {"dataset": "CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-YTTBAR"},
                  # {"dataset": "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR"}]

input_dict = dict(
    dataset_inputs=dataset_inputs,
    use_cuts="internal",
    metadata_group="nnpdf31_process",
    pdf="NNPDF40_nnlo_as_01180",
    theoryid=theoryid,
)

# toponium production cross section from arXiv:2102.11281
sigma_toponium_13tev = 6.43 # pb
sigma_toponium_8tev = 2.19 # pb

# predictions divided by binwidth
df = API.group_result_central_table_no_table(**input_dict)["theory_central"]


for dataset_input in dataset_inputs:
    dataset_name = dataset_input["dataset"]
    fk_table_path = fk_table_dir / f"theory_{theoryid}" / "fastkernel" / f"{dataset_name}.pineappl.lz4"
    grid = pineappl.grid.Grid.read(fk_table_path)
    dim = grid.bin_dimensions()
    bin_specs = np.array(grid.bin_limits())
    bin_width = grid.bin_normalizations()

    predictions = grid.convolve(
        pdg_convs=[conv_object, conv_object],  # Similar convolutions for symmetric protons
        xfxs=[pdf.xfxQ2, pdf.xfxQ2],  # Similar PDF sets for symmetric protons
        alphas=pdf.alphasQ2,
    )

    c_factors = [[1.0, 0.0] for _ in range(len(predictions))]


    xsec_threshold = predictions[0] * bin_width[0]

    if "13" in dataset_name:
        c_factor_toponium = (xsec_threshold + sigma_toponium_13tev) / xsec_threshold
    elif "8" in dataset_name:
        c_factor_toponium = (xsec_threshold + sigma_toponium_8tev) / xsec_threshold
    else:
        raise ValueError("Dataset energy not recognized in dataset name.")

    c_factors[0] = [c_factor_toponium, 0.0]

    header = f"""*******************************************************************************
SetName:     {dataset_name}
Author:      Jaco ter Hoeve jaco.ter.hoeve@ed.ac.uk
Date:        December 2025
CodesUsed: None 
TheoryInput: NNPDF4.0 alphas(mZ)=0.118
PDFset:      NNPDF40_nnlo_as_01180
Correcting for toponium production cross section from arXiv:2102.11281
********************************************************************************
"""

    with open(result_dir / f"CF_TOPO_{dataset_name}.dat", "w") as f:
        f.write(header)
        for bin_idx in range(len(predictions)):
            f.write(f"{c_factors[bin_idx][0]:.3f} {c_factors[bin_idx][1]:.3f}\n")

