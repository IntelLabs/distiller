#!/bin/bash

DATASET_DIR='../data/wmt16_de_en'

hiddens=(1024)
batches=(128)
maths=(fp16 fp32)
gpus=(1 2 4 8)

for hidden in "${hiddens[@]}"
do
   for math in "${maths[@]}"
   do
      for batch in "${batches[@]}"
      do
         for gpu in "${gpus[@]}"
         do
            export CUDA_VISIBLE_DEVICES=`seq -s "," 0 $((gpu - 1))`
            python3 -m multiproc train.py \
            --save benchmark_gpu_${gpu}_math_${math}_batch_${batch}_hidden_${hidden} \
            --dataset-dir ${DATASET_DIR} \
            --seed 12345 \
            --epochs 1 \
            --math ${math} \
            --print-freq 1 \
            --batch-size ${batch} \
            --disable-eval \
            --max-size $((128 * ${batch} * ${gpu})) \
            --max-length-train 48 \
            --max-length-val 150 \
            --model-config "{'num_layers': 8, 'hidden_size': ${hidden}, 'dropout':0.2, 'share_embedding': True}" \
            --optimization-config "{'optimizer': 'Adam', 'lr': 5e-4}"
         done
      done
   done
done
