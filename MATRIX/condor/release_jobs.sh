#!/bin/bash

# Command to get held jobs' ClusterId.ProcId
jobs=$(condor_q jthoeve -constraint 'JobStatus == 5' -af ClusterId)

# Loop over each job ID
for job in $jobs; do
    echo "Processing job $job"

#    # Example: Update attributes for the job
    condor_qedit "$job" RequestMemory 16384
#    condor_qedit "$job" JobCategory "long"

    # Release the job
    condor_release "$job"

    echo "Job $job updated and released"
done

#1: Idle
#2: Running
#3: Removed
#4: Completed
#5: Held
#6: Submission Error

# PUT ON HOLD

#jobs=$(condor_q jthoeve -constraint 'JobStatus == 1' -af ClusterId)
#
## Loop over each job ID
#for job in $jobs; do
#    echo "Processing job $job"
#
#    condor_hold "$job"
#
#    echo "Job $job put on hold"
#done
