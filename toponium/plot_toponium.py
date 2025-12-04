import matplotlib.pyplot as plt
import eventpy.event_reader as ep
import numpy as np
import pineappl
import lhapdf
import pandas as pd


from matplotlib import rc, use

use("PDF")
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})


def get_mtt(events_list):

    mtt_list = []
    weights = []


    for events in events_list:

        N = events.num_events

        for event in events.events:
            aux = 0

            for particle in event.intermediate_states:
                if particle.pdg == 6:
                    aux += particle.mom
                elif particle.pdg == -6:
                    aux += particle.mom
            if type(aux) is not int:
                mtt = np.sqrt(ep.dot(aux, aux))
                mtt_list.append(mtt)
                weights.append(event.weight / N)




    weights = np.array(weights)
    mtt_list = np.array(mtt_list)
    return (mtt_list, weights)

binL = 0.5
bins_mtt = np.arange(330, 350, binL)

mtt_events, weights = get_mtt([ep.Process("/Users/jaco/Documents/physics_projects/alphas_mtop/toponium/run_03/unweighted_events.lhe")])
mtt_binned, _ = np.histogram(mtt_events, bins=bins_mtt, weights=weights)
mtt_binned /= binL
mtt_binned *= 1e3  # fb/GeV
cos_CS_events_binned = np.append(mtt_binned, mtt_binned[-1])

fig, ax = plt.subplots(figsize=(8,6))
ax.step(bins_mtt, cos_CS_events_binned , where='post', label=r'$\mathrm{mg5\_aMC,\,NRQCD}$')

ax.set_xlabel(r"$W\;{\rm[GeV]}$")
ax.set_ylabel(r"$d\sigma\:\mathrm{[fb/GeV]}$")
#plt.yscale('log')
plt.grid()


ax.legend()

plt.savefig("toponium.pdf")






### LO DY plot was obtained with https://github.com/NNPDF/papers/blob/master/bsm_afb/FH_plots/plot.py










# plot analytic prediction for LO DY

