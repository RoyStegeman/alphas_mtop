import numpy as np
import pandas as pd
from validphys.api import API
import matplotlib.pyplot as plt

alphas_values=[0.116, 0.117, 0.118, 0.119, 0.120, 0.121, 0.122, 0.123]
mt_values=[170.0, 172.5, 175.0]

# alphas_values=[0.115]
# mt_values=[170.0]


fitname = f"250524-jth-exa-nnlo-mhou-mtt_alphas_0116_mt_170_0_iterated"

naive_dict = dict(
    fit=fitname,
    dataset_inputs={"from_": "fit"},
    pdf={"from_": "fit"},
    use_cuts="fromfit",
    theory={"from_": "fit"},
    theoryid={"from_": "theory"},
)

# t0 covariance matrix (the correct one, see bottom of page 15 of https://arxiv.org/pdf/1802.03398)
C = API.groups_covmat(
    fit=fitname,
    use_t0 = True,
    use_cuts="fromfit",
    datacuts={"from_": "fit"},
    t0pdfset={"from_": "datacuts"},
    dataset_inputs={"from_": "fit"},
    theoryid=API.fit(fit=fitname).as_input()["theory"]["t0theoryid"],
)

# the datapoint is already uniquely defined by the dataset and datapoint, we dont need the process
C = C.droplevel(0, axis=0).droplevel(0, axis=1)

stored_mhou_covmat = pd.read_csv(
    "/data/theorie/jthoeve/physics_projects/nnpdf_fits/results/250524-jth-exa-nnlo-mhou-mtt_alphas_0116_mt_170_0_iterated/tables/datacuts_theory_theorycovmatconfig_theory_covmat_custom.csv",
    index_col=[0, 1, 2],
    header=[0, 1, 2],
    sep="\t|,",
    encoding="utf-8",
    engine="python",
).fillna(0)

stored_mhou_covmat_index = pd.MultiIndex.from_tuples(
    [(aa, bb, np.int64(cc)) for aa, bb, cc in stored_mhou_covmat.index],
    names=["group", "dataset", "id"],
)

# make sure theoryID is an integer, same as in S
stored_mhou_covmat = pd.DataFrame(
    stored_mhou_covmat.values, index=stored_mhou_covmat_index, columns=stored_mhou_covmat_index
)

stored_mhou_covmat = stored_mhou_covmat.droplevel(0, axis=0).droplevel(0, axis=1)
stored_mhou_covmat = stored_mhou_covmat.reindex(C.index).T.reindex(C.index)

invcov = np.linalg.inv(C + stored_mhou_covmat)

results = []
print_header = True
for alphas_value in alphas_values:
    for mt_value in mt_values:
        print(f"Determining chi2 for alphas={alphas_value}, mt={mt_value}")

        alphas = "{:.3f}".format(alphas_value).replace('.', '')
        mt = str(mt_value).replace(".", "_")
        fitname = f"250524-jth-exa-nnlo-mhou-mtt_alphas_{alphas}_mt_{mt}_iterated"
        naive_dict["fit"] = fitname
        try:
            central_preds_and_data = API.group_result_central_table_no_table(**naive_dict)
        except:
            continue


        theory_db_id = API.fit(fit=fitname).as_input()["theory"]["theoryid"]
        alphas_from_fit = API.theory_info_table(theory_db_id=theory_db_id).loc["alphas"].iloc[0]
        mt_from_fit = API.theory_info_table(theory_db_id=theory_db_id).loc["mt"].iloc[0]

        assert alphas_from_fit == alphas_value, f"Expected alphas {alphas_value}, but got {alphas_from_fit} from fit {fitname}"
        assert mt_from_fit == mt_value, f"Expected mt {mt_value}, but got {mt_from_fit} from fit {fitname}"

        # compute chi2
        diff = central_preds_and_data.theory_central - central_preds_and_data.data_central
        
        chi2 = diff @ invcov @ diff / diff.size

        # Write the result to a file within the loop
        result_df = pd.DataFrame([[alphas_value, mt_value, chi2]], columns=['alpha_s', 'm_t', 'chi2'])
        result_df.to_csv(
            '/data/theorie/jthoeve/physics_projects/alphas_mtop/notebooks/results/250524-jth-exa-nnlo-mhou-mtt_alphas-exp-iterated.dat',
            sep='\t', index=False, mode='a', header=print_header)
        print_header = False


#
# alphas_values = np.array(alphas_values)
# mt_values = np.array(mt_values)
# chi2_values = np.array(chi2_values).reshape(-1, 3) # shape = (n as, n mt)
#
# ######## ALPHAS FIT
#
# a, b, c = np.polyfit(alphas_values, chi2_values[:, 1], 2)
#
# central = -b / 2 / a
# ndata = C.shape[0]
# unc = np.sqrt(1/a/ndata)
#
# plt.scatter(alphas_values, chi2_values, color="blue" )
# xgrid = np.linspace(min(alphas_values),max(alphas_values))
# plt.plot(xgrid, [a*x*x + b*x + c for x in xgrid], color="black", linestyle="--")
# plt.title(rf"$\alpha_s$={central:.5f}$\pm${unc:.5f}")
# print(f"{central:.5f} ± {unc:.5f}")
# plt.savefig("alpha_s_exp_mt_1725_8tev.png")
#
# ######## SIMULTANEOUS FIT
#
# # Construct the design matrix for quadratic terms
# X_matrix = np.column_stack([
#     alphas_values,        # x
#     alphas_values**2,     # x^2
#     mt_values,        # y
#     mt_values**2,     # y^2
#     alphas_values * mt_values,  # xy
#     np.ones_like(alphas_values)  # Constant term
# ])
# import pdb; pdb.set_trace()
# # Solve for [A, B, C, D, E, F] using least squares
# coeffs, _, _, _ = np.linalg.lstsq(X_matrix, chi2_values.flatten(), rcond=None)
# A_fit, B_fit, C_fit, D_fit, E_fit, F_fit = coeffs
#
# print(f"Fitted parameters: A={A_fit}, B={B_fit}, C={C_fit}, D={D_fit}, E={E_fit}, F={F_fit}")
# import pdb; pdb.set_trace()
# # Define the fitted quadratic function
# def quadratic_surface(x, y):
#     return (A_fit * x + B_fit * x**2 + C_fit * y + D_fit * y**2 + E_fit * x * y + F_fit)
#
# # Finding the intersection with z = z*
# z_star = 5.99  # Change this value as needed
# def intersection_curve(x, y):
#     return quadratic_surface(x, y) - z_star  # Solve for contour where this = 0
#
# # Create a grid for plotting
# x_vals = np.linspace(min(alphas_values), max(alphas_values), 100)
# y_vals = np.linspace(min(mt_values), max(mt_values), 100)
# X, Y = np.meshgrid(x_vals, y_vals)
# Z = intersection_curve(X, Y)
#
# # Plot the intersection curve
# plt.figure(figsize=(8, 6))
# contour = plt.contour(X, Y, Z, levels=[0], colors='r')
# import pdb; pdb.set_trace()
# # plt.scatter(al, y_data, c='b', label="Data Points")
# plt.xlabel("$\\alpha_s$")
# plt.ylabel("$m_t$")
# # plt.title(f"Intersection of Quadratic Surface with Plane z={z_star}")
# plt.legend()
# plt.savefig("intersection_curve.png")


