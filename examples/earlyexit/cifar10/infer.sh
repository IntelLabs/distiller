#!/bin/bash
for model in resnet20 resnet32 resnet44 resnet56 resnet110 resnet1202
do
  for threshold in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2
  do
    fullstring=$threshold.$model
    python -u trainer.py --workers=1 --evaluate --earlyexitmodel=save_1.5.$model/model.th --earlyexit=$threshold --arch=$model |& tee results_$fullstring
  done
done
