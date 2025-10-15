import tcm
import sys
import pathlib

fitname = sys.argv[1]
central_value, covmat = tcm.compute_posterior(fitname)
result_dir = pathlib.Path("/data/theorie/jthoeve/physics_projects/alphas_mtop/results/tcm_results")

# create directory fitname inside result_dir if it doesn't exist
(result_dir / fitname).mkdir(parents=True, exist_ok=True)

# save the central value and covariance matrix to a dat file
with open(result_dir / fitname / f"{fitname}_central_value.dat",'w') as f:
    f.write(f"{central_value}\n")

with open(result_dir / fitname / f"{fitname}_covmat.dat",'w') as f:
    for row in covmat:
        f.write(" ".join(map(str, row)) + "\n")

print(f"Results saved in {result_dir / fitname}")