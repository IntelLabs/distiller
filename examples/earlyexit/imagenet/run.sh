#!/bin/bash
export TORCH_HOME=/tmp
#d=$(date +%Y-%m-%d)
threshold0=1.2
threshold1=1.4
lossweight0=0.1
lossweight1=0.3
fullstring="${lossweight0}_${lossweight1}"
fullstring2="${threshold0}_${threshold1}"

echo "python -u main.py --pretrained --batch-size=64 --earlyexit $threshold0 $threshold1 --arch=resnet50 --lossweights $lossweight0 $lossweight1 --checkpointdir=$fullstring --lr=0.01 /dataset/aeon/I1K/i1k-extracted/ |& tee -a $fullstring/log_$fullstring2"
python -u main.py --pretrained --batch-size=64 --earlyexit $threshold0 $threshold1 --arch=resnet50 --lossweights $lossweight0 $lossweight1 --checkpointdir=$fullstring --lr=0.01 /dataset/aeon/I1K/i1k-extracted/ |& tee -a $fullstring/log_$fullstring2