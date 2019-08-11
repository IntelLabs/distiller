# Object Detection Compression

In this example we compress torchvision's object detection models - FasterRCNN / MaskRCNN / KeypointRCNN.  
We've modified the [reference code for object detection](https://github.com/pytorch/vision/tree/master/references/detection)
to allow easy compression scheduling with yaml configuration.

## Running the Example
The command line for running this example is closely related to 
[`compress_classifier.py`](distiller/examples/classifier_compression/compress_classifier.py)