# Object Detection Compression

In this example we compress torchvision's object detection models - FasterRCNN / MaskRCNN / KeypointRCNN.  
We've modified the [reference code for object detection](https://github.com/pytorch/vision/tree/master/references/detection)
to allow easy compression scheduling with yaml configuration.

## Setup
Install the dependencies 
(most of which are already installed from Distiller dependies, the rest are `Cython` and `pycocotools`):
    
    cd <distiller root>/examples/object_detection_compression/
    pip3 install -r requirements.txt 

The dataset can be downloaded at the [COCO dataset website](http://cocodataset.org/#download).
Please keep in mind that COCO dataset takes up 18 GB of storage.  
In this example we'll use the 2017 training+validation sets, which you can download using the command line:

    cd data
    bash download_dataset.sh

## Running the Example
The command line for running this example is closely related to 
[`compress_classifier.py`](../classifier_compression/compress_classifier.py), i.e. the
compression scheduler format and most of the Distiller related arguments are the same.  
However - running in a multi-GPU environment is different from `compress_classifier.py`, because this script is a modified
[`train.py` from torchvision references](https://github.com/pytorch/vision/tree/master/references/detection/train.py), 
where they used `torch.distributed.launch` for multi-GPU (and multi-node in general) training.

**Note-** Use of `torch.distributed.launch` will spawn multiple processes, on each process
there will be a copy of the model and the weights, each of the models is an instance of 
[`torch.nn.parallel.DistributedDataParallel`](https://pytorch.org/docs/stable/nn.html#distributeddataparallel).
During backward pass, the gradients from each node are averaged and then passed to all nodes,
thus promising the weights on the nodes are the same. 
This also promises that the pruning masks remain identical on all the nodes.
 
 Example Single GPU Command Line - 
 
    python compress_detector.py --data-path /path/to/dataset.COCO --pretrained --compress maskrcnn.scheduler_agp.non_parallel.yaml

 Example Multi GPU Command Line -  
 
    python -m torch.distributed.launch --nproc_per_node=4 --use_env compress_detector.py --data-path /path/to/dataset.COCO \
     --compress maskrcnn.scheduler_agp.yaml --pretrained --world-size 4 --batch-size 2 --epochs 80

Since the dataset is large and FasterRCNN models are compute heavy, we strongly recommend
running the script on a Multi GPU environment. Keep in mind that the multi-GPU case is 
running on multiple processes via `torch.distributed.launch`, and ending one of the processes
might break all of them and leave them in an undefined state (In that case you'll have to end 
them manually). Also, even though the multi-GPU distributes the memory over all the GPUs, the 
model is quite memory intensive, so using a large batch size is guaranteed to yield OOM on the GPU. 
Our GPUs are TITAN X (Pascal) with 12GB memory, and a batch size of 3 is the most we could run without 
memory errors. 


The default model is `torchvision.models.detection.maskrcnn_resnet50_fpn`, you can specify 
any model that is part of `torchvision.models.detection` using
 the `--model` argument, e.g. `--model maskrcnn_resnet50_fpn`.