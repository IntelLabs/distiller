#!/bin/bash
export TORCH_HOME=/tmp
model=resnet50
lossweight0=0.1
lossweight1=0.3
fullstring="${lossweight0}_${lossweight1}/model_best.pth.tar"
for threshold0 in 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4
    do
    for threshold1 in 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0
        do
            fullstring2="${threshold0}_${threshold1}"
            logstring="${lossweight0}_${lossweight1}/results_${threshold0}_${threshold1}"
            python -u main.py --workers=1 --batch-size=1 --evaluate --earlyexitmodel=$fullstring --earlyexit $threshold0 $threshold1 --arch=$model /dataset/aeon/I1K/i1k-extracted/ |& tee $logstring
    done
done