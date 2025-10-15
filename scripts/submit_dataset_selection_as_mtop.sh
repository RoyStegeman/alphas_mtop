#!/bin/bash

# condor_submit script for dataset selection alphas mtop

# Define variables for the directory containing runcards
FITS_DIR="/data/theorie/jthoeve/physics_projects/nnpdf_share/results"

FITS=()
# loop over all directories in FITS_DIR that start with 250926-jth-dataset-selection-with
for FIT in "$FITS_DIR"/251014-jth-dataset-selection-with*/; do
  [ -d "$FIT" ] || continue
  FITS+=( "$(basename "$FIT")" )
done


# Define the wrapper script for running smefit
WRAPPER_SCRIPT="run_tcm.sh"

# Create the wrapper script
cat <<EOL > $WRAPPER_SCRIPT
#!/bin/bash
# Wrapper script for executing smefit SCAN

# Get the runcard file from the arguments
FIT=\$1

python /data/theorie/jthoeve/physics_projects/alphas_mtop/scripts/run_tcm.py \$FIT
EOL

# Make the wrapper script executable
chmod +x $WRAPPER_SCRIPT

# Create the HTCondor submit description file
cat <<EOL > submit_tcm_alphas_mtop.submit
# HTCondor submit file

universe = vanilla
executable = $WRAPPER_SCRIPT

# Transfer input files (the runcards)
arguments = \$(Item)

+UseOS                  = "el9"
+JobCategory            = "short"
request_cpus   = 1
request_memory = 8G
getenv = true
accounting_group = smefit

# Define log, output, and error files
log = logs/smefit_scan_\$(Cluster)_\$(Process).log
output = logs/smefit_scan_\$(Cluster)_\$(Process).out
error = logs/smefit_scan_\$(Cluster)_\$(Process).err

# Queue jobs for each runcard
queue Item from (
EOL


# Append runcard list to the submit file
for FIT in "${FITS[@]}"; do
  echo "$FIT" >> submit_tcm_alphas_mtop.submit
done

echo ")" >> submit_tcm_alphas_mtop.submit

# Submit the jobs
#condor_submit submit_file_scan.submit
