#!/bin/bash

PINEAPPLE=/project/theorie/jthoeve/miniconda3/envs/pinefarm/bin/pineappl

OUTPUT_DIR="/data/theorie/jthoeve/physics_projects/matrix-alle-hawaii2/MATRIX/run/ppttx20_MATRIX/input"
for grid in ./*.lz4; do

  if [[ -f $grid ]]; then
    
    dataset_name=$(basename "$grid" .pineappl.lz4)
    if [[ "$dataset_name" == *"TTBAR"* && "$dataset_name" != *"INTEGRATED"* ]]; then
      mkdir -p $OUTPUT_DIR/$dataset_name
      $PINEAPPLE read --get dddistribution.dat $grid >> $OUTPUT_DIR/$dataset_name/dddistribution.dat
      $PINEAPPLE read --get distribution.dat $grid >> $OUTPUT_DIR/$dataset_name/distribution.dat
      $PINEAPPLE read --get model.dat $grid >> $OUTPUT_DIR/$dataset_name/model.dat
      $PINEAPPLE read --get parameter.dat $grid >> $OUTPUT_DIR/$dataset_name/parameter.dat
    fi
  fi
done


