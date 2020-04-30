#!/usr/bin/env bash

# This script was used to generate the results included in
# <distiller_root>/examples/quantization/fp32_baselines/preact_resnet_cifar_base_fp32.yaml
# and
# <distiller_root>/examples/quantization/quant_aware_train/preact_resnet_cifar_dorefa.yaml
#
# IMPORTANT: 
#  * It is assumed that the script is executed from the following directory:
#     <distiller_root>/examples/classifier_compression
#    Some of the paths used are relative to this directory.
#  * Some of the paths might need to be modified for your own system, e.g. 'dataset' and 't_ckpt'

gpu=$1
suffix=$2

if [ -z $gpu ]; then
  gpu=0
fi

if [ -z $suffix ]; then
  suffix="try1"
fi

# Modify dataset path to your own
dataset="$MACHINE_HOME/datasets/cifar10"
base_fp32_sched="../quantization/fp32_baselines/preact_resnet_cifar_base_fp32.yaml"
dorefa_w3_a8_sched="../quantization/quant_aware_train/preact_resnet_cifar_dorefa.yaml"
base_out_dir="logs/presnet_cifar"

base_args="${dataset} --lr 0.1 -p 50 -b 128 -j 1 --epochs 200 --wd 0.0002 --vs 0 --gpus ${gpu}"
base_cmd="python compress_classifier.py"

# No distillation
for mode in "base_fp32" "dorefa_w3_a8"; do
  for depth in 20 32 44 56 110; do
    arch=preact_resnet${depth}_cifar
    sched=${mode}_sched
    out_dir="${base_out_dir}/presnet${depth}"
    exp_name="presnet${depth}_${mode}_${suffix}"
    set -x
    ${base_cmd} -a ${arch} ${base_args} --compress=${!sched} -o ${out_dir} -n ${exp_name}
    set +x
  done
done

# With distillation
for mode in "base_fp32" "dorefa_w3_a8"; do
  sched=${mode}_sched
  for s_depth in 20 32 44 56; do
    for temp in 1 2 5; do
      for dw in 0.7; do
        for sw in 0.3; do
          for t_depth in 32 44 56 110; do
            if (( $t_depth > $s_depth )); then
              s_arch=preact_resnet${s_depth}_cifar
              t_arch=preact_resnet${t_depth}_cifar
			  # Change t_ckpt path to point to your pre-trained checkpoints
              t_ckpt="../baselines/models/${t_arch}10/checkpoint_best.pth.tar"
              out_dir="${base_out_dir}/presnet${s_depth}"
              exp_name="presnet${s_depth}_${t_depth}_t_${temp}_dw_${dw}_${mode}_${suffix}"
              kd_args="--kd-teacher ${t_arch} --kd-resume ${t_ckpt} --kd-temp ${temp} --kd-dw ${dw} --kd-sw ${sw}"
              set -x
              ${base_cmd} -a ${s_arch} ${base_args} --compress=${!sched} ${kd_args} -o ${out_dir} -n ${exp_name}
              set +x
            fi
          done
        done
      done
    done
  done
done
