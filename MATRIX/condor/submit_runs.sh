#!/bin/bash


INPUT_DIR=/data/theorie/jthoeve/physics_projects/matrix-alle-hawaii2/MATRIX/run/ppttx20_MATRIX/input/
HTCONDOR_THRESHOLD=100
MASSES_MT=("172.5" "170.0" "175.0")

check_htcondor_jobs() {
    # Get the number of jobs in the HTCondor queue
    current_jobs=$(condor_q -totals $USER | grep "Total for query" | awk '{print $4}')
    echo $current_jobs
}

# loop over datasets
for PROCESS_DIR in $INPUT_DIR/*; do

    PROCESS=$(basename "$PROCESS_DIR")

    # skip non-ttbar processes
    [[ "$PROCESS" != *"TTBAR"* || "$PROCESS" == *"run"* ]] && continue

    # loop over masses
    index=0
    for MT in "${MASSES_MT[@]}"; do

      MODEL_FILE=$PROCESS_DIR/"model.dat"

      # the first mass run includes the full run
      if [[ $index -eq 0 ]]; then
        ./bin/run_process "run_"$PROCESS --run_mode run --input_dir $PROCESS

        # uncomment option after first mass run to avoid including the pre-in results in the subsequent mass runs
        sed -i '/^#include_pre_in_results = 0/s/^#//' $INPUT_DIR/"run_"$PROCESS/parameter.dat
      else
        # update the top mass
        sed -i "s/^\(\s*6\s*\)[^ ]*/\1$MT/" "$MODEL_FILE"

        # run the main stage only. We run in the background and continue to the next run if the number of jobs is
        # below the threshold
        ./bin/run_process "run_"$PROCESS --run_mode run_main &

        # wait until the number of jobs is below the threshold
        while true; do
           num_current_jobs=$(check_htcondor_jobs)
           # if the number of jobs is below the threshold, break the loop
           if [[ $num_current_jobs -lt $HTCONDOR_THRESHOLD ]]; then
              break
           fi
          sleep 900  # Check every 15 min
        done
      fi

      if [[ $? -ne 0 ]]; then
          echo "Command failed on $PROCESS and $MT Exiting loop."
          break  # Exit the loop if the command failed
      fi
      ((index++))

    done
#  break
done


# maybe useful
# --run_mode run_pre_and_main to skip the generation of the grids

# to continue a run
# ./bin/run_process run_ATLAS_TTBAR_13TEV_HADR_DIF_mt_172P5 --run_mode run_main --continue  --input_dir ATLAS_TTBAR_13TEV_HADR_DIF