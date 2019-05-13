#!/bin/bash

set -e

DATASET_DIR='../data/wmt16_de_en'
RESULTS_DIR='gnmt_wmt16'

# evaluate best checkpoint on newstest2014
python3 -u translate.py \
   --math fp16 \
   --model ../results/${RESULTS_DIR}/model_best.pth \
   --input ${DATASET_DIR}/newstest2014.tok.bpe.32000.en \
   --output ../results/${RESULTS_DIR}/newstest2014_out.tok.de \
   |tee ../results/${RESULTS_DIR}/inference.log

# detokenize output
perl ${DATASET_DIR}/mosesdecoder/scripts/tokenizer/detokenizer.perl -l de \
   < ../results/${RESULTS_DIR}/newstest2014_out.tok.de \
   > ../results/${RESULTS_DIR}/newstest2014_out.de

# compute uncased BLEU
cat ../results/${RESULTS_DIR}/newstest2014_out.de \
   |sacrebleu ${DATASET_DIR}/newstest2014.de -lc \
   --tokenize intl |tee ../results/${RESULTS_DIR}/bleu_nt14_lc.log

# compute cased BLEU
cat ../results/${RESULTS_DIR}/newstest2014_out.de \
   |sacrebleu ${DATASET_DIR}/newstest2014.de \
   --tokenize intl |tee ../results/${RESULTS_DIR}/bleu_nt14.log
