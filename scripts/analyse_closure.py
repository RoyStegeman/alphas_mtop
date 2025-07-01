import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 16})
rc('text', usetex=True)

as_values = []
as_unc = []
mt_values = []
mt_unc = []
for i in range(2, 101):
    file_cv = f"./results/closure_tests/250624-{i + 1:03d}-jth-closuretest-alphas-mtop-nopos-nodiag_iterated_iterated_central_value.txt"
    try:
        with open(pathlib.Path(file_cv), "r") as f:
            lines = f.readlines()
            values = [float(entry) for entry in lines[0].split('\t')]

            mt_values.append(values[0])
            as_values.append(values[1])
    except FileNotFoundError:
        continue

    file_unc = f"./results/closure_tests/250624-{i + 1:03d}-jth-closuretest-alphas-mtop-nopos-nodiag_iterated_iterated_covmat.txt"
    try:
        with open(pathlib.Path(file_unc), "r") as f:
            lines = f.readlines()
            covmat = []
            for line in lines:
                covmat.append([float(entry) for entry in line.split('\t')])
            covmat = np.array(covmat)
            mt_unc.append(np.sqrt(covmat[0, 0]))
            as_unc.append(np.sqrt(covmat[1, 1]))
    except FileNotFoundError:
        continue

as_values = np.array(as_values)
mt_values = np.array(mt_values)
as_unc = np.array(as_unc)
mt_unc = np.array(mt_unc)

weights_as = 1 / as_unc**2
weights_mt = 1 / mt_unc**2
# compute weighted mean of mt and as
mt_mean = np.sum(mt_values * weights_mt) / np.sum(weights_mt)
as_mean = np.sum(as_values * weights_as) / np.sum(weights_as)

# compute standard deviation of mt and as
mt_std = 1 / np.sqrt(np.sum(weights_mt))
as_std = 1 / np.sqrt(np.sum(weights_as))

# compute standard deviation of the mean




# plot histograms

fig, ax = plt.subplots(figsize=(10, 6))
plt.hist(as_values, bins='fd', alpha=0.5, label=r'$\alpha_s$', ec='k')
plt.xlabel(r'$\alpha_s$')
fig.savefig("./results/plots/250626-jth-closuretest-nodiag-alphas-iterated-iterated.png")
fig, ax = plt.subplots(figsize=(10, 6))
plt.hist(mt_values, bins='fd', alpha=0.5, label=r'$m_t$', ec='k')
plt.xlabel(r'$m_t$')
fig.savefig("./results/plots/250626-jth-closuretest-nodiag-mt-iterated-iterated.png")


fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(mt_values, as_values)
ax.set_xlabel(r'$m_t\:{\rm [GeV]}$', fontsize=25)
ax.set_ylabel(r'$\alpha_s$', fontsize=25)
ax.axvline(172.5, color='k', linestyle='--')
ax.axvline(mt_mean, color='C1', linestyle='--')
ax.axhline(0.118, color='k', linestyle='--', label=r'${\rm True\:value}$')
ax.axhline(as_mean, color='C1', linestyle='--', label=r'${\rm Mean\:value}$')

mt_min = mt_values.min() - 0.05 * (mt_values.max() - mt_values.min())
mt_max = mt_values.max() + 0.05 * (mt_values.max() - mt_values.min())
as_min = as_values.min() - 0.05 * (as_values.max() - as_values.min())
as_max = as_values.max() + 0.05 * (as_values.max() - as_values.min())
mt_range = np.linspace(mt_min, mt_max, 100)
as_range = np.linspace(as_min, as_max, 100)
ax.fill_between(mt_range, as_mean - as_std, as_mean + as_std, color='C1', alpha=0.2)
ax.fill_betweenx(as_range, mt_mean - mt_std, mt_mean + mt_std, color='C1', alpha=0.2)
ax.legend(frameon=False)
ax.set_xlim(mt_min, mt_max)
ax.set_ylim(as_min, as_max)
ax.set_title(r'${\rm TCM\:without\:positivity}$')
fig.savefig("./results/plots/250626-jth-closuretest-nodiag-mt-as-iterated_iterated.pdf")

