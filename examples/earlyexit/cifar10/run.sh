#!/bin/bash
threshold=1.5
for model in resnet20 resnet32 resnet44 resnet56 resnet110 resnet1202
do
    fullstring=$threshold.$model
    python -u trainer.py --workers=1 --pretrained --earlyexit=$threshold \
        --arch=$model --save-dir=save_$fullstring |& tee -a log_$fullstring
done