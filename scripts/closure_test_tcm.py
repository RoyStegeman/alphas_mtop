import sys
import tcm

# take fitname for first argument passed to this script

num = int(sys.argv[1])
fitname = f"250630-{num:03d}-jth-closuretest-alphas-mtop-nopos-nodiag_iterated_iterated"


central_value, covmat = tcm.compute_posterior(fitname)


# save the central value and covariance matrix to a file
with open(f"./results/closure_tests/{fitname}_central_value.txt", "w") as f:
    f.write("\t".join(map(str, central_value)))

with open(f"./results/closure_tests/{fitname}_covmat.txt", "w") as f:
    for row in covmat:
        f.write("\t".join(map(str, row)) + "\n")