# AMC: AutoML for Model Compression and Acceleration on Mobile Devices

## Notebook examples of AMC applied to compress CNNs:
- [Random](./jupyter/amc_random.ipynb)
- [Plain20](./jupyter/amc_plain20.ipynb)
- [Resnet20](./jupyter/amc_resnet20.ipynb)
- [Resnet 56]()
- [Mobilenet v1]()
- [Mobilenet v2]()


| Model | Baseline Top1 | AMC paper | AMC Distiller
| --- |  ---: |  ---: |  ---: |
| Plain20 | 90.5 |
| Resnet20 | 91.78 |
| Resnet56 | 92.85 |
| Mobilenet v1 |  |
| Mobilenet v2 |  |


AMC: AutoML for Model Compression and Acceleration on Mobile Devices.<br>
     Yihui He, Ji Lin, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han.<br>
     In Proceedings of the European Conference on Computer Vision (ECCV), 2018.<br>
     [arXiv:1802.03494](https://arxiv.org/abs/1802.03494)<br>

We thank Prof. Song Han and his team for their [help](https://github.com/mit-han-lab/amc-compressed-models) with certain critical parts of this implementation.  However, all bugs in interpretation and/or implementation are ours ;-).
