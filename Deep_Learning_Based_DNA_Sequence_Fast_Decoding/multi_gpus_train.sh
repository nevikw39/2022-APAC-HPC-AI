#!/bin/bash

# change the path to your own directory
path="/scratch/jx00/cw2590/DL-based-DNA-decoding"
output_path="$path/output"

## To train a single model with two gpus
horovodrun -np $PBS_NGPUS python3 $path/deep_tf.py -m leopard_unet -b 64
# horovodrun -np $PBS_NGPUS python3 $path/deep_tf.py -m cnn_more_dense

## To train multiple models
# array=( leopard_unet )
# for i in "${array[@]}"
# do
# 	horovodrun -np $PBS_NGPUS python3 "$path"/deep_tf.py -m $i -b 64
# done
