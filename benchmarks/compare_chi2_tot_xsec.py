# This script reproduces the chi2 of the tot xsec in https://vp.nnpdf.science/Ms_vTGFuRWeh2x6td-AoHw==/#chi-data (NNLO)
# and https://vp.nnpdf.science/F2hM3Gg6Q1u-E9jFsqIKAQ==/#chi-data (NLO + K factor)

from validphys.api import API
import numpy as np

ds_inputs = [
     "ATLAS_TTBAR_7TEV_TOT_X-SEC",
]

nlo_k_ds_inps = [{"dataset": ds, "variant": "legacy_theory"} for ds in ds_inputs]
nnlo_ds_inps = [{"dataset": ds} for ds in ds_inputs]


dict_nlo_k = dict(theoryid=708, dataset_inputs=nlo_k_ds_inps, pdf="NNPDF40_nnlo_as_01180", use_cuts="internal")
dict_nnlo = dict(theoryid=40009000, dataset_inputs=nnlo_ds_inps, pdf="NNPDF40_nnlo_as_01180", use_cuts="internal")

ds_info_nlo_k = API.group_result_central_table_no_table(**dict_nlo_k)
preds_nlo_k = ds_info_nlo_k["theory_central"]
data_central = ds_info_nlo_k["data_central"]
exp_covmat = API.groups_covmat(
    use_t0=False,
    t0pdfset="NNPDF40_nnlo_as_01180",
    dataset_inputs=nlo_k_ds_inps,
    theoryid=708,
    use_cuts="internal",
)
exp_covmat = exp_covmat.reindex(preds_nlo_k.index).T.reindex(preds_nlo_k.index)
ndat = data_central.size
chi2_nlo_k = (data_central - preds_nlo_k).T @ np.linalg.inv(exp_covmat) @ (data_central - preds_nlo_k) / ndat

### NNLO

ds_info_nnlo = API.group_result_central_table_no_table(**dict_nnlo)
preds_nnlo = ds_info_nnlo["theory_central"]
exp_covmat_nnlo = API.groups_covmat(
    use_t0=False,
    t0pdfset="NNPDF40_nnlo_as_01180",
    dataset_inputs=nnlo_ds_inps,
    theoryid=708,
    use_cuts="internal",
)
exp_covmat_nnlo = exp_covmat_nnlo.reindex(preds_nnlo.index).T.reindex(preds_nnlo.index)
chi2_nnlo = (data_central - preds_nnlo).T @ np.linalg.inv(exp_covmat) @ (data_central - preds_nnlo) / ndat

import pdb; pdb.set_trace()
