#!/bin/bash


INPUT_DIR=/data/theorie/jthoeve/physics_projects/matrix-alle-hawaii2/MATRIX/run/ppttx20_MATRIX/input
HTCONDOR_THRESHOLD=100
#MASSES_MT=("172.5" "170.0" "175.0")
MASSES_MT=("170.0" "175.0")
#PROCESSES=("ATLAS_TTBAR_13TEV_LJ_DIF" "ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR" "ATLAS_TTBAR_8TEV_DIF" "CMS_TTBAR_13TEV_2L_DIF" "CMS_TTBAR_5TEV_TOT_X-SEC" "CMS_TTBAR_7TEV_TOT_X-SEC" "CMS_TTBAR_8TEV_2L_DIF")
PROCESSES=("CMS_TTBAR_7TEV_TOT_X-SEC" "CMS_TTBAR_5TEV_TOT_X-SEC")
#PROCESSES=("CMS_TTBAR_7TEV_TOT_X-SEC")
RUNPHASES=("run_grid" "run_pre" "run_main")

check_htcondor_jobs() {
    # Get the number of jobs in the HTCondor queue
    current_jobs=$(condor_q -totals $USER | grep "Total for query" | awk '{print $4}')
    echo $current_jobs
}

delay_until_space() {
  # checks whether the number of jobs in the queue is below the threshold before moving to the next run
  while true; do
     sleep 120  # Check every 15 min
     num_current_jobs=$(check_htcondor_jobs)
     # if the number of jobs is below the threshold, break the loop
     if [[ $num_current_jobs -lt $HTCONDOR_THRESHOLD ]]; then
        break  # break out of the while loop
     fi
     echo $num_current_jobs " jobs in the queue, waiting till this is below " $HTCONDOR_THRESHOLD

  done

  return 0
}

counter=0
pids=()  # Array to store PIDs of the background processes
for MT in "${MASSES_MT[@]}"; do
  for PHASE in "${RUNPHASES[@]}"; do
    for PROCESS in "${PROCESSES[@]}"; do
      echo "Starting process $PROCESS in phase $PHASE"

      PROCESS_DIR=$INPUT_DIR/$PROCESS

      if [[ $counter -gt 0 ]]; then
        MODEL_FILE=$INPUT_DIR/"run_"$PROCESS/"model.dat"
        sed -i "s/^\(\s*6\s*\)[^ ]*/\1$MT/" "$MODEL_FILE"
      fi

      # grid run (if not done already)
      if [[ $PHASE == "run_grid" && $counter -eq 0 ]]; then
        # set the top mass
        MODEL_FILE=$INPUT_DIR/$PROCESS/"model.dat"
        sed -i "s/^\(\s*6\s*\)[^ ]*/\1$MT/" "$MODEL_FILE"

        ./bin/run_process "run_"$PROCESS --run_mode run_grid --input_dir $PROCESS &
        pids+=($!)
        echo "during grid run: "${pids[@]}
        delay_until_space  # only continue from here if the number of jobs in the queue is below the threshold
      # pre run (if not done already)
      elif [[ $PHASE == "run_pre" && $counter -eq 0 ]]; then
        # First check whether all background jobs from the previous phase (run_grid) have finished
        echo "before staring pre-run: "${pids[@]}
        for pid in "${pids[@]}"; do
            wait $pid  # Wait for each process in the array
            echo "Command with PID $pid has finished"
        done
        pids=()  # reset the array of PIDs
        echo "All grid runs have finished. Continuing to the pre stage."

        # check if main.running is gone
        while true; do
          if [[ ! -f $PWD/log/run_$PROCESS/main.running ]]; then break; fi
        done

        # start the pre-run
        ./bin/run_process "run_"$PROCESS --run_mode run_pre &
        pids+=($!)
        delay_until_space  # only continue from here if the number of jobs in the queue is below the threshold
      # main run
      elif [[ $PHASE == "run_main" ]]; then

         # First check whether all background jobs from the previous phase (run_pre) have finished
        for pid in "${pids[@]}"; do
            wait $pid  # Wait for each process in the array
            echo "Command with PID $pid has finished"
        done
        pids=()  # reset the array of PIDs
        echo "All pre-runs have finished. Continuing to the main stage."

        # check if main.running is gone
        while true; do
          if [[ ! -f $PWD/log/run_$PROCESS/main.running ]]; then break; fi
        done

        # uncomment include_pre_in_results for the main run after the first mass point
        if [[ $counter -gt 0 ]]; then
          sed -i '/^#include_pre_in_results = 0/s/^#//' $INPUT_DIR/"run_"$PROCESS/parameter.dat
        fi

        ./bin/run_process "run_"$PROCESS --run_mode run_main &
        pids+=($!)
        delay_until_space  # only continue from here if the number of jobs in the queue is below the threshold

      fi
    done
  done
  counter=$((counter + 1))  # Increment counter
done

# maybe useful
# --run_mode run_pre_and_main to skip the generation of the grids

# to continue a run
# ./bin/run_process run_ATLAS_TTBAR_13TEV_HADR_DIF_mt_172P5 --run_mode run_main --continue  --input_dir ATLAS_TTBAR_13TEV_HADR_DIF