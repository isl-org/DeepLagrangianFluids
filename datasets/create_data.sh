#!/bin/bash

# Set the path to SPlishSPlasHs DynamicBoundarySimulator in splishsplash_config.py
# before running this script

# output directories
OUTPUT_SCENES_DIR=ours_default_scenes
OUTPUT_DATA_DIR=ours_default_data

mkdir $OUTPUT_SCENES_DIR

# This script is purely sequential but it is recommended to parallelize the
# following loop, which generates the simulation data.
for seed in `seq 1 220`; do
        python create_physics_scenes.py --output $OUTPUT_SCENES_DIR \
                                        --seed $seed \
                                        --default-viscosity \
                                        --default-density
done


# Transforms and compresses the data such that it can be used for training.
# This will also create the OUTPUT_DATA_DIR.
python create_physics_records.py --input $OUTPUT_SCENES_DIR \
                                 --output $OUTPUT_DATA_DIR 


# Split data in train and validation set
mkdir $OUTPUT_DATA_DIR/train
mkdir $OUTPUT_DATA_DIR/valid

for seed in `seq -w 1 200`; do
        mv $OUTPUT_DATA_DIR/sim_0${seed}_*.msgpack.zst $OUTPUT_DATA_DIR/train
done

for seed in `seq -w 201 220`; do
        mv $OUTPUT_DATA_DIR/sim_0${seed}_*.msgpack.zst $OUTPUT_DATA_DIR/valid
done
