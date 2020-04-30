#!/bin/bash

# This script was used to generate the results shown in post-training quantization command line readme at:
# <distiller_root>/examples/quantization/post_train_quant/command_line.md
# Note that the readme shows only a subset of the tests run by this script, especially the 6-bits tests.
# This scripts also runs the conversion to "native" PyTorch post-train quant feature.
#
# IMPORTANT: 
#  * It is assumed that the script is run from the following directory:
#     <distiller_root>/examples/classifier_compression
#    Some of the paths used are relative to this directory.

model="resnet50"
dataset_path="$MACHINE_HOME/datasets/imagenet"
stats_file="../quantization/post_train_quant/stats/${model}_quant_stats.yaml"
out_dir="logs/${model}_ptq_pytorch_convert_latest"
num_workers=22

base_args="--arch ${model} ${dataset_path} --pretrained -j ${num_workers} --evaluate --quantize-eval --qe-stats-file ${stats_file} -o ${out_dir}"

for engine in distiller pytorch; do
  convert_flag=""
  if [ "$engine" = pytorch ]; then
    convert_flag="--qe-convert-pytorch"
  fi
  for n_bits in 8 6; do
    for acts_mode in sym asym_u; do
      for wts_mode in sym asym_u; do
        for per_ch in per_tensor per_channel; do
          per_ch_flag=""
          if [ "$per_ch" = per_channel ]; then
            per_ch_flag="--qe-per-channel"
          fi
          exp_name="acts_${acts_mode}_wts_${wts_mode}_${per_ch}_${engine}"
          set -x
          python compress_classifier.py ${base_args} --qe-mode-acts ${acts_mode} --qe-mode-wts ${wts_mode} --qe-bits-acts ${n_bits} --qe-bits-wts ${n_bits} ${per_ch_flag} ${convert_flag} --name ${exp_name}
          set +x
        done
      done
    done
  done
done
