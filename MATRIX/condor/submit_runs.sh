#!/bin/bash


INPUT_DIR=/data/theorie/jthoeve/physics_projects/matrix-alle-hawaii2/MATRIX/run/ppttx20_MATRIX/input
HTCONDOR_THRESHOLD=100
MASSES_MT=("172.5" "170.0" "175.0")
PROCESSES=("ATLAS_TTBAR_13TEV_LJ_DIF" "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR" "ATLAS_TTBAR_8TEV_DIF" "CMS_TTBAR_13TEV_2L_DIF" "CMS_TTBAR_5TEV_TOT_X-SEC" "CMS_TTBAR_7TEV_TOT_X-SEC" "CMS_TTBAR_8TEV_2L_DIF")

check_htcondor_jobs() {
    # Get the number of jobs in the HTCondor queue
    current_jobs=$(condor_q -totals $USER | grep "Total for query" | awk '{print $4}')
    echo $current_jobs
}

delay_until_space() {
  # checks whether the number of jobs in the queue is below the threshold before moving to the next run
  while true; do
     sleep 900  # Check every 15 min
     num_current_jobs=$(check_htcondor_jobs)
     # if the number of jobs is below the threshold, break the loop
     if [[ $num_current_jobs -lt $HTCONDOR_THRESHOLD ]]; then
        break  # break out of the while loop
     fi
     echo $num_current_jobs " jobs in the queue, waiting till this is below " $HTCONDOR_THRESHOLD

  done

  return 0
}


# loop over mass
for MT in "${MASSES_MT[@]}"; do

    pids=()  # Array to store PIDs of the background processes
    echo "Starting mass run for mt = $MT"

    for PROCESS in "${PROCESSES[@]}"; do
      echo "Starting process $PROCESS"

      PROCESS_DIR=$INPUT_DIR/$PROCESS
      MODEL_FILE=$INPUT_DIR/"run_"$PROCESS/"model.dat"

      # the first mass run includes the full run
      if [[ $MT == "172.5" && "$PROCESS" == "ATLAS_TTBAR_13TEV_LJ_DIF" ]]; then
        ./bin/run_process "run_"$PROCESS --run_mode run_pre_and_main --input_dir $PROCESS &
        pids+=($!)
        delay_until_space
      elif [[ $MT == "172.5" ]]; then
        ./bin/run_process "run_"$PROCESS --run_mode run --input_dir $PROCESS &
        pids+=($!)
        delay_until_space
      else
        # update the top mass
        sed -i "s/^\(\s*6\s*\)[^ ]*/\1$MT/" "$MODEL_FILE"

        # run the main stage only (we already have a successful run).
        ./bin/run_process "run_"$PROCESS --run_mode run_main &
        pids+=($!)  # store the process id
        delay_until_space
      fi
    done

    # Wait for all background jobs to finish before moving to the next mass point
    for pid in "${pids[@]}"; do
        wait $pid  # Wait for each process in the array
        echo "Command with PID $pid has finished"
    done
    echo "All processes for mass $MT have finished"
    echo "Uncommenting the pre-in results for the next mass run"

    # uncomment the pre-in results for the next mass run
    for PROCESS in "${PROCESSES[@]}"; do
      sed -i '/^#include_pre_in_results = 0/s/^#//' $INPUT_DIR/"run_"$PROCESS/parameter.dat
    done

done


# maybe useful
# --run_mode run_pre_and_main to skip the generation of the grids

# to continue a run
# ./bin/run_process run_ATLAS_TTBAR_13TEV_HADR_DIF_mt_172P5 --run_mode run_main --continue  --input_dir ATLAS_TTBAR_13TEV_HADR_DIF