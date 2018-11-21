# Conditional Computation

Conditional Computation refers to a class of algorithms in which each input sample uses a different part of the model, such that on average the compute, latency or power (depending on our objective) is reduced.
To quote [Bengio et. al](#bengio)
> "Conditional computation refers to activating only some of the units in a network, in an input-dependent fashion. For example, if we think we’re looking at a car, we only need to compute the activations of the vehicle detecting units, not of all features that a network could possible compute. The immediate effect of activating fewer units is that propagating information through the network will be faster, both at training as well as at test time. However, one needs to be able to decide in an intelligent fashion which units to turn on and off, depending on the input data. This is typically achieved with some form of gating structure, learned in parallel with the original network."

As usual, there are several approaches to implement Conditional Computation:

* [Sun et. al](#sun) use several expert CNN, each trained on a different task, and combine them to one large network.
* [Zheng et. al](#zheng) use cascading, an idea which may be familiar to you from Viola-Jones face detection.
* [Theodorakopoulos et. al](#theodorakopoulos) add small layers that learn which filters to use per input sample, and then enforce that during inference (LKAM module).
* [Ioannou et. al](#ioannou) introduce Conditional Networks: that "can be thought of as: i) decision trees augmented with data transformation
operators, or ii) CNNs, with block-diagonal sparse weight matrices, and explicit data routing functions"
* [Bolukbasi et. al](#bolukbasi) "learn a system to adaptively choose the components of a deep network to be evaluated for each example. By allowing examples correctly classified using early layers of the system to exit, we avoid the computational time associated with full evaluation of the network. We extend this to learn a network selection system that adaptively selects the network to be evaluated for each example."

Conditional Computation is especially useful for real-time, latency-sensitive applicative.<br>
In Distiller we currently have implemented a variant of Early Exit.


## References
<div id="bengio"></div> **Emmanuel Bengio, Pierre-Luc Bacon, Joelle Pineau, Doina Precup.**
    [*Conditional Deep Learning for Energy-Efficient and Enhanced Pattern Recognition*](https://arxiv.org/abs/1511.06297), arXiv:1511.06297v2, 2016.

<div id="sun"></div>**Y. Sun, X.Wang, and X. Tang.**
    *Deep Convolutional Network Cascade for Facial Point Detection*. In Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2014

<div id="zheng"></div>**X. Zheng, W.Ouyang, and X.Wang.** *Multi-Stage Contextual Deep Learning for Pedestrian Detection.* In Proc. IEEE Intl Conf. on Computer Vision (ICCV), 2014.

<div id="theodorakopoulos"></div>**I. Theodorakopoulos, V. Pothos, D. Kastaniotis and N. Fragoulis1.** *Parsimonious Inference on Convolutional Neural Networks: Learning and applying on-line kernel activation rules.* Irida Labs S.A, January 2017

<div id="bolukbasi"></div>**Tolga Bolukbasi, Joseph Wang, Ofer Dekel, Venkatesh Saligrama** [*Adaptive Neural Networks for Efficient Inference*](http://proceedings.mlr.press/v70/bolukbasi17a/bolukbasi17a.pdf).  Proceedings of the 34th International Conference on Machine Learning, PMLR 70:527-536, 2017.

<div id="ioannou"></div> **Yani Ioannou, Duncan Robertson, Darko Zikic, Peter Kontschieder, Jamie Shotton, Matthew Brown, Antonio Criminisi**.
    [*Decision Forests, Convolutional Networks and the Models in-Between*](https://arxiv.org/abs/1511.06297), arXiv:1511.06297v2, 2016.
