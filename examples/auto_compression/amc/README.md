# AMC: AutoML for Model Compression and Acceleration on Mobile Devices

## Notebook examples of AMC applied to compress CNNs:
- [Random](./jupyter/amc_random.ipynb)
- [Plain20](./jupyter/amc_plain20.ipynb)
- [Resnet20](./jupyter/amc_resnet20.ipynb)
- [Resnet 56]()
- [Mobilenet v1]()
- [Mobilenet v2]()


| Model | FLOPs | Baseline Top1 | AMC paper<br>(Top1 after FT)  | Baseline Top1 | AMC Distiller
| --- |  ---: |  ---: |  ---: |  ---: |  ---: |
| Plain20 | 50% | 90.5 | 90.2 (-0.3) | 90.55 | 90.04 (-0.4)
| Resnet20 | 50%| 91.78 | - | 
| Resnet56 | 50% |92.8 | 91.9 (-0.9)
| Mobilenet v1 | 50% | 70.6 | 70.2 (-0.4)
| Mobilenet v2 |  50% | 71.8 | 70.8 (-1.0)

These notebooks will help visualize and review the results of recreating the results of AMC.

AMC trains a Deep Deterministic Policy Gradient (DDPG) RL agent to compress DNNs. For the implementation of DDPG we're using [Coach](https://github.com/NervanaSystems/coach) or Prof. Han's team's [implementation](https://github.com/mit-han-lab/amc-compressed-models) . We wrap Distiller in an OpenAI [Gym environment API](https://github.com/openai/gym) so that it presents a standard RL environment to the agent. We execute Distiller's sample application [amc.py](https://github.com/NervanaSystems/distiller/blob/new_amc/examples/auto_compression/amc/amc.py) with special arguments telling it how to invoke AMC. This creates a DistillerWrapperEnvironment environment and a DDPG agent and starts the training cycle. For the exact technical details of doing this yourself, see the notebooks (better instructions will be added in the future).

These notebooks only review Resource-Constrained Compression.

AMC: AutoML for Model Compression and Acceleration on Mobile Devices.<br>
     Yihui He, Ji Lin, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han.<br>
     In Proceedings of the European Conference on Computer Vision (ECCV), 2018.<br>
     [arXiv:1802.03494](https://arxiv.org/abs/1802.03494)<br>

We thank Prof. Song Han and his team for their [help](https://github.com/mit-han-lab/amc-compressed-models) with certain critical parts of this implementation.  However, all bugs in interpretation and/or implementation are ours ;-).
