#!/bin/bash

set -e

DATASET_DIR='../data/wmt16_de_en'
RESULTS_DIR='gnmt_wmt16'

# sort by length (ascending)
cat ${DATASET_DIR}/newstest2014.tok.bpe.32000.en \
   | awk '{ print length, $0 }' \
   | sort -n -s \
   | cut -d" " -f2- > /tmp/newstest2014.tok.bpe.32000.en.sorted

batches=(512 256 128 64 32)
beams=(1 2 5 10)
maths=(fp16 fp32)

model=../results/${RESULTS_DIR}/model_best.pth

odir=../results/inference_benchmark
mkdir -p $odir

echo RUNNING on unsorted dataset
rm -rf $odir/fp16_perf_unsorted.log
rm -rf $odir/fp32_perf_unsorted.log
rm -rf $odir/fp16_bleu.log
rm -rf $odir/fp32_bleu.log
ifile=${DATASET_DIR}/newstest2014.tok.bpe.32000.en

for math in "${maths[@]}"
do
   for batch in "${batches[@]}"
   do
      for beam in "${beams[@]}"
      do
         echo RUNNING: batch_size: $batch, beam_size: $beam, math: $math

         # run translation
         python3 translate.py \
            -i $ifile \
            -m $model \
            --math $math \
            --print-freq 1 \
            --beam-size $beam \
            --batch-size $batch \
            -o /tmp/output.tok &> /tmp/log.log

         tok_per_sec=`cat /tmp/log.log \
            |tail -n 1 \
            |cut -f 1 \
            |cut -d ':' -f 2 |tr -d ' '`

         # detokenize output
         perl ${DATASET_DIR}/mosesdecoder/scripts/tokenizer/detokenizer.perl -l de \
            < /tmp/output.tok \
            > /tmp/output.detok 2> /dev/null

         bleu=`sacrebleu ${DATASET_DIR}/newstest2014.de --input /tmp/output.detok -lc --tokenize intl -b`

         echo -e $tok_per_sec '\t\t batch: '$batch 'beam: ' $beam >> $odir/${math}_perf_unsorted.log
         echo -e $bleu '\t\t batch: '$batch 'beam: ' $beam  >> $odir/${math}_bleu.log
         echo Tokens per second: $tok_per_sec, BLEU: $bleu
      done
   done
done


echo RUNNING on sorted dataset
rm -rf $odir/fp16_perf_sorted.log
rm -rf $odir/fp32_perf_sorted.log
ifile=/tmp/newstest2014.tok.bpe.32000.en.sorted


for math in "${maths[@]}"
do
   for batch in "${batches[@]}"
   do
      for beam in "${beams[@]}"
      do
         echo RUNNING: batch_size: $batch, beam_size: $beam, math: $math

         # run translation
         python3 translate.py \
            -i $ifile \
            -m $model \
            --math $math \
            --print-freq 1 \
            --beam-size $beam \
            --batch-size $batch \
            -o /tmp/output.tok &> /tmp/log.log

         tok_per_sec=`cat /tmp/log.log \
            |tail -n 1 \
            |cut -f 1 \
            |cut -d ':' -f 2 |tr -d ' '`

         echo -e $tok_per_sec '\t\t batch: '$batch 'beam: ' $beam >> $odir/${math}_perf_sorted.log
         echo Tokens per second: $tok_per_sec
      done
   done
done
