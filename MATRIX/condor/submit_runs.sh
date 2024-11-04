#!/bin/bash


INPUT_DIR=/data/theorie/jthoeve/physics_projects/matrix-alle-hawaii2/MATRIX/run/ppttx20_MATRIX/input/

MASSES_MT=("172.5" "170.0" "175.0")
# loop over datasets
for PROCESS_DIR in $INPUT_DIR/*; do

    PROCESS=$(basename "$PROCESS_DIR")

    # loop over masses
    index=0
    for MT in "${MASSES_MT[@]}"; do
      if [[ "$PROCESS" == *"TTBAR"* && "$PROCESS" != *"run" ]]; then

        MODEL_FILE=$INPUT_DIR/"run_"$PROCESS/"model.dat"

        # set up the run
        if [[ $index -eq 0 ]]; then
          ./bin/run_process "run_"$PROCESS --run_mode run --input_dir $PROCESS

          # uncomment option after first mass run
          sed -i '/^#include_pre_in_results = 0/s/^#//' $INPUT_DIR/"run_"$PROCESS/parameter.dat
        else
          # update the top mass
          sed -i "s/^\(\s*6\s*\)[^ ]*/\1$MT/" "$MODEL_FILE"

          # only run the main stage
          ./bin/run_process "run_"$PROCESS --run_mode run_main
        fi

        if [[ $? -ne 0 ]]; then
            echo "Command failed on $PROCESS and $MT Exiting loop."
            break  # Exit the loop if the command failed
        fi
        ((index++))
      fi
    done
#  break
done


# maybe useful
# --run_mode run_pre_and_main to skip the generation of the grids

# to continue a run
# ./bin/run_process run_ATLAS_TTBAR_13TEV_HADR_DIF_mt_172P5 --run_mode run_main --continue  --input_dir ATLAS_TTBAR_13TEV_HADR_DIF