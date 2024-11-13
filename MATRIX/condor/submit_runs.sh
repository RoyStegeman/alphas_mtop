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


# GRID RUN

pids=()  # Array to store PIDs of the background processes
PHASE="run_grid"
for PROCESS in "${PROCESSES[@]}"; do

  echo "Starting process $PROCESS in phase $PHASE"

  # set the initial top mass
  MODEL_FILE=$INPUT_DIR/$PROCESS/"model.dat"
  sed -i "s/^\(\s*6\s*\)[^ ]*/\1${MASSES_MT[0]}/" "$MODEL_FILE"

  PROCESS_DIR=$INPUT_DIR/$PROCESS

  # grid run (if not done already)
  if [[ $PHASE == "run_grid" ]]; then
    ./bin/run_process "run_"$PROCESS --run_mode run_grid --input_dir $PROCESS &
    pids+=($!)
    delay_until_space  # only continue from here if the number of jobs in the queue is below the threshold
  fi
done

# MAKE SURE TO WAIT FOR THE GRID RUNS TO FINISH BEFORE CONTINUING
echo "before staring pre-run: "${pids[@]}
for pid in "${pids[@]}"; do
    wait $pid  # Wait for each process in the array
    echo "Command with PID $pid has finished"
done
pids=()  # reset the array of PIDs
echo "All grid runs have finished"
echo "checking if main.running is gone"

for PROCESS in "${PROCESSES[@]}"; do
  while true; do
    if [[ ! -f $PWD/log/run_$PROCESS/main.running ]]; then break; fi
  done
done
echo "Continuing to the pre stage."

# PRE-RUN
PHASE="run_pre"

for PROCESS in "${PROCESSES[@]}"; do
  echo "Starting process $PROCESS in phase $PHASE"

  PROCESS_DIR=$INPUT_DIR/$PROCESS

  if [[ $PHASE == "run_pre" ]]; then

    # check if main.running is gone
    while true; do
      if [[ ! -f $PWD/log/run_$PROCESS/main.running ]]; then break; fi
    done

    # start the pre-run
    ./bin/run_process "run_"$PROCESS --run_mode run_pre &
    pids+=($!)
    delay_until_space  # only continue from here if the number of jobs in the queue is below the threshold
  fi
done

# MAKE SURE TO WAIT FOR THE GRID RUNS TO FINISH BEFORE CONTINUING
echo "Processes that need to finish before being able to start the main run: "${pids[@]}
for pid in "${pids[@]}"; do
    wait $pid  # Wait for each process in the array
    echo "Command with PID $pid has finished"
done
pids=()  # reset the array of PIDs
echo "All pre runs have finished"
echo "checking if main.running is gone"
for PROCESS in "${PROCESSES[@]}"; do
  while true; do
    if [[ ! -f $PWD/log/run_$PROCESS/main.running ]]; then break; fi
  done
done
echo "Continuing to the main stage."

# MAIN RUN
counter=0
for MT in "${MASSES_MT[@]}"; do

  for PROCESS in "${PROCESSES[@]}"; do

    # set the top mass
    MODEL_FILE=$INPUT_DIR/"run_"$PROCESS/"model.dat"
    sed -i "s/^\(\s*6\s*\)[^ ]*/\1$MT/" "$MODEL_FILE"

    if [[ $counter -gt 0 ]]; then
      sed -i '/^#include_pre_in_results = 0/s/^#//' $INPUT_DIR/"run_"$PROCESS/parameter.dat
    fi
    ./bin/run_process "run_"$PROCESS --run_mode run_main &
    pids+=($!)
    delay_until_space
  done

  # before moving to the next iteration of mt, wait for all the main runs to finish
  for pid in "${pids[@]}"; do
    wait $pid  # Wait for each process in the array
    echo "Command with PID $pid has finished"
  done
  pids=()  # reset the array of PIDs

  counter=$((counter+1))
done



# maybe useful
# --run_mode run_pre_and_main to skip the generation of the grids

# to continue a run
# ./bin/run_process run_ATLAS_TTBAR_13TEV_HADR_DIF_mt_172P5 --run_mode run_main --continue  --input_dir ATLAS_TTBAR_13TEV_HADR_DIF