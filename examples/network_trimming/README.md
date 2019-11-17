## Network Trimming Pruning

### Examples
In theses example schedules, after pruning the filters, we remove them ("thinning") and continue fine-tuning.

| Model | Granularity | Sparsity (%) | Parameters Kept (%) | Compute Kept (%)| Top1 | Baseline Top1
| --- |  :--- |  ---: |  ---: |  ---: | ---: |  ---: |
| ResNet-50 | Filters| 0.0  | 43.37 | 44.56 | 73.93 | 76.15
| ResNet-56 | Filters| 0.0  | 74.53 | 62.71 | 93.03 | 92.85
| ResNet-56 | Filters| 0.0  | 67.02 | 53.92 | 92.59 | 92.85
