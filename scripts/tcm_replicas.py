import numpy as np
import pandas as pd
from validphys.api import API

import matplotlib.pyplot as plt
from matplotlib import patches, transforms, rc
from matplotlib.patches import Ellipse

import scipy
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})

def compute_posterior(fitname):
    fit = API.fit(fit=fitname)

    # We have to know the name of the alphas and mtop point prescriptions (alphas_pp and mtop_pp) to
    # extract the theoryids. We need alphas_pp_id and mtop_pp_id to identify the .csv file
    # corresponding to the alphas covmat and mtop used in the fit.
    pps = fit.as_input()["theorycovmatconfig"]["point_prescriptions"]
    for i, pp in enumerate(pps):
        if "mtop" in pp:
            mtop_pp_id = i
            mtop_pp = pp
        elif "alphas" in pp:
            alphas_pp_id = i
            alphas_pp = pp
        else:
            continue

    common_dict = dict(
        dataset_inputs={"from_": "fit"},
        fit=fit.name,
        fits=[fit.name],
        use_cuts="fromfit",
        metadata_group="nnpdf31_process",
    )

    theoryids_dict_mtop = ({
                               "point_prescription": mtop_pp,
                               "theoryid": {"from_": "theory"},
                               "theory": {"from_": "fit"},
                               "theorycovmatconfig": {"from_": "fit"},
                           } | common_dict)

    theoryids_dict_alphas = ({
                                 "point_prescription": alphas_pp,
                                 "theoryid": {"from_": "theory"},
                                 "theory": {"from_": "fit"},
                                 "theorycovmatconfig": {"from_": "fit"},
                             } | common_dict)

    # extract theory ids for mtop and alphas including their variations
    theoryids_mtop = API.theoryids(**theoryids_dict_mtop)
    theoryids_alphas = API.theoryids(**theoryids_dict_alphas)


    theory_central = theoryids_mtop[0].id
    theory_plus = [theoryids_mtop[2].id, theoryids_alphas[1].id]
    theory_min = [theoryids_mtop[1].id, theoryids_alphas[2].id]

    thcov_input_pdf = fit.as_input()["theorycovmatconfig"]["pdf"]

    # Inputs for central theory (used to construct the mtop, alphas covmat).
    inps_central = dict(theoryid=theory_central, pdf=thcov_input_pdf, **common_dict)

    # Inputs for plus theory (used to construct the mtop, alphas covmat)
    inps_plus = [dict(theoryid=theory_id_plus, pdf=thcov_input_pdf, **common_dict) for theory_id_plus in theory_plus]

    # Inputs for minus theory prediction (used to construct the mtop, alphas covmat)
    inps_min = [dict(theoryid=theory_id_min, pdf=thcov_input_pdf, **common_dict) for theory_id_min in theory_min]

    # inputs for the computation of the prediction of the fit with cov=C+S, where S
    # is computed using the inps_central, inps_plus, and inps_minus dictionaries
    inps_central_fit = dict(theoryid=theory_central, pdf={"from_": "fit"}, **common_dict)
    
    # Get the prior theory predictions for the central values
    prior_theorypreds_central = API.group_result_central_table_no_table(**inps_central)[
        "theory_central"].to_frame()  # shape (n_dat, 1)

    # Get the prior theory predictions for the plus and minus variations
    prior_theorypreds_plus = pd.concat(
        [API.group_result_central_table_no_table(**inp_plus)["theory_central"] for inp_plus in inps_plus],
        axis=1)  # shape (n_dat, n_par)
    prior_theorypreds_minus = pd.concat(
        [API.group_result_central_table_no_table(**inp_min)["theory_central"] for inp_min in inps_min],
        axis=1)  # shape (n_dat, n_par)

    # Get the values of mto
    mtop_plus = API.theory_info_table(theory_db_id=theory_plus[0]).loc["mt"].iloc[0]
    mtop_central = API.theory_info_table(theory_db_id=theory_central).loc["mt"].iloc[0]
    mtop_min = API.theory_info_table(theory_db_id=theory_min[0]).loc["mt"].iloc[0]

    # and alphas
    alphas_plus = API.theory_info_table(theory_db_id=theory_plus[1]).loc["alphas"].iloc[0]
    alphas_central = API.theory_info_table(theory_db_id=theory_central).loc["alphas"].iloc[0]
    alphas_min = API.theory_info_table(theory_db_id=theory_min[1]).loc["alphas"].iloc[0]

    # and make sure the shift in both directions is symmetric
    delta_plus = np.array([mtop_plus - mtop_central, alphas_plus - alphas_central])
    delta_min = np.array([mtop_central - mtop_min, alphas_central - alphas_min])
    
    if np.any(abs(delta_min - delta_plus) > 1e-6):
        raise ValueError("mtop shifts in both directions is not symmetric")
    else:
        step_size = np.array(delta_min).reshape(-1, 1)

    # At some point we scaled the covmat to account for higher order derivatives or
    # to test depencence of the prior. It is not used in the final result
    covmat_scaling_factor = 1  # fit.as_input().get("theorycovmatconfig",{}).get("rescale_alphas_covmat",1.0)

    # Compute theory covmat S_tilde on the genuine predictions (as, mt)
    beta_tilde = np.sqrt(covmat_scaling_factor) * (step_size / np.sqrt(2)) * np.array([[1, -1, 0, 0], [0, 0, 1, -1]])
    S_tilde = beta_tilde @ beta_tilde.T

    # TODO: why a npsqrt(2)
    delta_plus = (np.sqrt(covmat_scaling_factor) / np.sqrt(2)) * (
            prior_theorypreds_plus - prior_theorypreds_central
    )
    delta_minus = (np.sqrt(covmat_scaling_factor) / np.sqrt(2)) * (
            prior_theorypreds_minus - prior_theorypreds_central
    )

    # Compute the theory cross covmat between the fitted predictions and the genuine predictions
    beta = np.array([delta_plus.iloc[:, 0].values, delta_minus.iloc[:, 0].values, delta_plus.iloc[:, 1].values,
                     delta_minus.iloc[:, 1].values]).T  # shape (n_dat, 2 * n_par)
    S_hat = beta_tilde @ beta.T  # shape (n_par, n_dat)

    # Compute the theory covmat on the theory predictions
    S = beta @ beta.T
    S = pd.DataFrame(S, index=delta_minus.index, columns=delta_minus.index)

    stored_alphas_covmat = pd.read_csv(
        fit.path / f"tables/datacuts_theory_theorycovmatconfig_point_prescriptions{alphas_pp_id}_theory_covmat_custom_per_prescription.csv",
        index_col=[0, 1, 2],
        header=[0, 1, 2],
        sep="\t|,",
        encoding="utf-8",
        engine="python",
    ).fillna(0)

    stored_mtop_covmat = pd.read_csv(
        fit.path / f"tables/datacuts_theory_theorycovmatconfig_point_prescriptions{mtop_pp_id}_theory_covmat_custom_per_prescription.csv",
        index_col=[0, 1, 2],
        header=[0, 1, 2],
        sep="\t|,",
        encoding="utf-8",
        engine="python",
    ).fillna(0)

    stored_covmat = stored_alphas_covmat + stored_mtop_covmat

    storedcovmat_index = pd.MultiIndex.from_tuples(
        [(aa, bb, np.int64(cc)) for aa, bb, cc in stored_covmat.index],
        names=["group", "dataset", "id"],
    )

    # make sure theoryID is an integer, same as in S
    stored_covmat = pd.DataFrame(
        stored_covmat.values, index=storedcovmat_index, columns=storedcovmat_index
    )
    stored_covmat = stored_covmat.reindex(S.index).T.reindex(S.index)

    if not np.allclose(S, stored_covmat):
        print("Reconstructed theory covmat, S, is not the same as the stored covmat!")

    data_theory_results = API.group_result_table_no_table(**inps_central_fit)
    theorypreds_fit = data_theory_results.iloc[:, 2:]

    # experimental covmat
    C = API.groups_covmat(
        use_t0=True,
        datacuts={"from_": "fit"},
        t0pdfset={"from_": "datacuts"},
        theoryid={"from_": "theory"},
        theory={"from_": "fit"},
        **common_dict
    )

    # %%
    # Note that mean_prediction is different from the prediction of the mean PDF (i.e. replica0)
    T0 = theorypreds_fit.mean(axis=1)

    X = np.cov(theorypreds_fit, bias=True)


    # In the computation we use <D>_rep and not the central value of the data D_exp, though if
    # resample_negative_pseudodata: false
    # is set in the n3fit runcard, D_exp and <D>_rep should be the same as N_rep -> inf.
    pseudodata = API.read_pdf_pseudodata(**common_dict)
    dat_reps = pd.concat(
        [i.pseudodata.reindex(prior_theorypreds_central.index) for i in pseudodata], axis=1
    )
    dat_central = dat_reps.mean(axis=1)



    invcov = np.linalg.inv(C + S)

    delta_T_tilde_reps = -S_hat @ invcov @ (theorypreds_fit.values - dat_central.values.reshape(-1, 1))
    delta_T_tilde = -S_hat @ invcov @ (T0 - dat_central)

    central_theory = np.array([mtop_central, alphas_central])

    # P_tilde is Eq. 3.38.
    #
    # Note that not all terms of the equation in the paper are here, in particular
    # X_tile and X_hat vanish. This is because they measure the covariance of
    # T_tilde over PDF replicas, but for us T_tilde is alphas. The prediciton of
    # alphas does not depend on the PDF, and as such T_tilde^(r) == T_tilde^(0)
    P_tilde = S_hat @ invcov @ X @ invcov @ S_hat.T + S_tilde - S_hat @ invcov @ S_hat.T
    pred = central_theory + delta_T_tilde
    pred_reps = central_theory.reshape(-1, 1) + delta_T_tilde_reps

    plot_histograms(pred, P_tilde, pred_reps, np.cov(pred_reps))

    return pred, P_tilde

def plot_histograms(pred, P_tilde, pred_reps, P_tilde_reps):

    # mtop: replicas vs formula

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pred_reps[0, :], bins="fd",color="C0",edgecolor="black", alpha=0.3)
    ax.axvline(x=pred[0], color='C1', linestyle='--')
    ax.axvline(x=pred[0] + np.sqrt(P_tilde[0,0]), color='C1', linestyle=':')
    ax.axvline(x=pred[0] - np.sqrt(P_tilde[0,0]), color='C1', linestyle=':', label="Formula")
    ax.axvline(x=np.mean(pred_reps[0, :]), color='C2', linestyle='--')
    ax.axvline(x=np.mean(pred_reps[0, :]) - np.sqrt(P_tilde_reps[0,0]), color='C2', linestyle=':')
    ax.axvline(x=np.mean(pred_reps[0, :]) + np.sqrt(P_tilde_reps[0,0]), color='C2', linestyle=':', label="Replicas")
    plt.legend()
    fig.savefig("./results/plots/mt_reps_hist.png")

    # alphas: replicas vs formula
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pred_reps[1, :], bins="fd", color="C0", edgecolor="black", alpha=0.3)
    ax.axvline(x=pred[1], color='C1', linestyle='--')
    ax.axvline(x=pred[1] + np.sqrt(P_tilde[1, 1]), color='C1', linestyle=':')
    ax.axvline(x=pred[1] - np.sqrt(P_tilde[1, 1]), color='C1', linestyle=':', label="Formula")
    ax.axvline(x=np.mean(pred_reps[1, :]), color='C2', linestyle='--')
    ax.axvline(x=np.mean(pred_reps[1, :]) - np.sqrt(P_tilde_reps[1, 1]), color='C2', linestyle=':')
    ax.axvline(x=np.mean(pred_reps[1, :]) + np.sqrt(P_tilde_reps[1, 1]), color='C2', linestyle=':', label="Replicas")
    plt.legend()
    fig.savefig("./results/plots/as_reps_hist.png")

def confidence_ellipse(
        cov, mean, ax, facecolor="none", confidence_level=95, **kwargs
):
    """
    Draws 95% C.L. ellipse for data points `x` and `y`

    Parameters
    ----------
    coeff1: array_like
        ``(N,) ndarray`` containing ``N`` posterior samples for the first coefficient
    coeff2: array_like
        ``(N,) ndarray`` containing ``N`` posterior samples for the first coefficient
    ax: matplotlib.axes
        Axes object to plot on
    facecolor: str, optional
        Color of the ellipse
    **kwargs
        Additional plotting settings passed to matplotlib.patches.Ellipse

    Returns
    -------
    matplotlib.patches.Ellipse
        Ellipse object

    """
    # diagonalise
    eig_val, eig_vec = np.linalg.eig(cov)

    # eigenvector with largest eigenvalue
    eig_vec_max = eig_vec[:, np.argmax(eig_val)]

    # angle of eigenvector with largest eigenvalue with the horizontal axis
    cos_th = eig_vec[0, np.argmax(eig_val)] / np.linalg.norm(eig_vec_max)
    if eig_vec_max[1] > 0:
        inclination = np.arccos(cos_th)
    else:
        # pay attention to range of arccos (extend to [0, -\pi] domain)
        inclination = -np.arccos(cos_th)

    eigval_sort = np.sort(eig_val)

    chi2_qnt = scipy.stats.chi2.ppf(confidence_level / 100.0, 2)

    ell_radius_x = np.sqrt(chi2_qnt * eigval_sort[-1])
    ell_radius_y = np.sqrt(chi2_qnt * eigval_sort[-2])

    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )

    mean_coeff1 = mean[0]
    mean_coeff2 = mean[1]

    transf = (
        transforms.Affine2D().rotate(inclination).translate(mean_coeff1, mean_coeff2)
    )

    # ax.set_xlim(pred[0] - ell_radius_x * 1.5, pred[0] + ell_radius_x * 1.5)
    # ax.set_ylim(pred[1] - ell_radius_y * 1.5, pred[1] + ell_radius_y * 1.5)

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)

fitname = "250630-100-jth-closuretest-alphas-mtop-nopos-nodiag_iterated_iterated"
central_values, _ = compute_posterior(fitname)