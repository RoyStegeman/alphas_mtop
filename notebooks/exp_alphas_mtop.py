import numpy as np
import pandas as pd
from validphys.api import API
import matplotlib.pyplot as plt

alphas_values=["0114", "0115", "0116", "0117", "0118", "0119", "0120", "0121", "0122", "0123", "0124", "0125"]
mt_values=["170_0", "172_5", "175_0"]

# alphas_values=[ "0115", "0116", "0117", "0118", "0119", "0120", "0121", "0122", "0123", "0124"]
# mt_values=["175_0"]

fit_names = []
for alphas in alphas_values:
    for mt in mt_values:
        fit_names.append(f"250325-jth-exa-nnlo_alphas_{alphas}_mt_{mt}")


#%%
naive_dict = dict(
    fit=fit_names[0],
    dataset_inputs={"from_": "fit"},
    pdf={"from_": "fit"},
    use_cuts="fromfit",
    theory={"from_": "fit"},
    theoryid={"from_": "theory"},
)

# t0 covariance matrix (the correct one, see bottom of page 15 of https://arxiv.org/pdf/1802.03398)
C = API.groups_covmat(
    fit=fit_names[0],
    use_t0 = True,
    use_cuts="fromfit",
    datacuts={"from_": "fit"},
    t0pdfset={"from_": "datacuts"},
    dataset_inputs={"from_": "fit"},
    theoryid=API.fit(fit=fit_names[0]).as_input()["theory"]["t0theoryid"],
)

# the datapoint is already uniquely defined by the dataset and datapoint, we dont need the process
C = C.droplevel(0, axis=0).droplevel(0, axis=1)

invcov = np.linalg.inv(C)

chi2_values = []
alphas_values = []
mt_values = []

for fitname in fit_names:
    print(fitname)
    naive_dict["fit"] = fitname
    central_preds_and_data = API.group_result_central_table_no_table(**naive_dict)

    # TODO: check that data is the same for all as

    theory_db_id = API.fit(fit=fitname).as_input()["theory"]["theoryid"]
    alphas_values.append(API.theory_info_table(theory_db_id=theory_db_id).loc["alphas"].iloc[0])
    mt_values.append(API.theory_info_table(theory_db_id=theory_db_id).loc["mt"].iloc[0])

    # compute chi2
    diff = central_preds_and_data.theory_central - central_preds_and_data.data_central
    chi2_values.append(diff @ invcov @ diff / diff.size)
    print(chi2_values)

alphas_values = np.array(alphas_values)
mt_values = np.array(mt_values)
chi2_values = np.array(chi2_values)

######## ALPHAS FIT

# a, b, c = np.polyfit(alphas_values, chi2_values, 2)
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

######## SIMULTANEOUS FIT

# Construct the design matrix for quadratic terms
X_matrix = np.column_stack([
    alphas_values,        # x
    alphas_values**2,     # x^2
    mt_values,        # y
    mt_values**2,     # y^2
    alphas_values * mt_values,  # xy
    np.ones_like(alphas_values)  # Constant term
])

# Solve for [A, B, C, D, E, F] using least squares
coeffs, _, _, _ = np.linalg.lstsq(X_matrix, chi2_values, rcond=None)
A_fit, B_fit, C_fit, D_fit, E_fit, F_fit = coeffs

print(f"Fitted parameters: A={A_fit}, B={B_fit}, C={C_fit}, D={D_fit}, E={E_fit}, F={F_fit}")
import pdb; pdb.set_trace()
# Define the fitted quadratic function
def quadratic_surface(x, y):
    return (A_fit * x + B_fit * x**2 + C_fit * y + D_fit * y**2 + E_fit * x * y + F_fit)

# Finding the intersection with z = z*
z_star = 5.99  # Change this value as needed
def intersection_curve(x, y):
    return quadratic_surface(x, y) - z_star  # Solve for contour where this = 0

# Create a grid for plotting
x_vals = np.linspace(min(alphas_values), max(alphas_values), 100)
y_vals = np.linspace(min(mt_values), max(mt_values), 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = intersection_curve(X, Y)

# Plot the intersection curve
plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=[0], colors='r')
import pdb; pdb.set_trace()
# plt.scatter(al, y_data, c='b', label="Data Points")
plt.xlabel("$\\alpha_s$")
plt.ylabel("$m_t$")
# plt.title(f"Intersection of Quadratic Surface with Plane z={z_star}")
plt.legend()
plt.savefig("intersection_curve.png")


