import distiller
import torch
import torch.optim
import torch.nn as nn
import distiller.apputils.image_classifier as classifier
import logging

from examples.classifier_compression import parser
from distiller.data_loggers import *
from distiller.models import create_model
from distiller.apputils import image_classifier
from distiller.apputils import load_data
# from distiller.models.cifar10.vgg_cifar import vgg16_bn_cifar  # distiller model
from models.vgg16_bn import vgg16_bn_cifar_custom                # custom model

def main():
    
    ### set arguments
    args = parser.add_cmdline_args(classifier.init_classifier_compression_arg_parser(True)).parse_args()
    args.device = 'cuda'
    args.print_freq = 50
    args.batch_size = 128
    
    
    ### make train, valid, test data loader
    loader = load_data('cifar10', '../../../data.cifar10', batch_size=args.batch_size, workers=1)
    (train_loader, val_loader, test_loader) = (loader[0], loader[1], loader[2])
    
    
    ### load model
    # model from distiller
    model = create_model(False, 'cifar10', 'preact_resnet20_cifar',
                         parallel=not False, device_ids=None)
    
    # model from custom
    # model = vgg16_bn_cifar_custom()
    
    if not isinstance(model, nn.DataParallel) :
        model.features = nn.DataParallel(model.features)
        model.is_parallel = True
        model.to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=0.0002)
    print(model)
    
    
    # convert model using PACT Quantization
    yaml_path = '/home/dwkim/Project/distiller/examples/quantization/quant_aware_train/preact_resnet20_cifar_pact.yaml'    
    compression_scheduler = None
    compression_scheduler = distiller.file_config(model, optimizer, yaml_path, compression_scheduler, None)
    print(model)
    
    # set loss function
    criterion = nn.CrossEntropyLoss()
    
    # Logger handle
    msglogger = logging.getLogger()
    logdir = image_classifier._init_logger(args, '/home/dwkim/Project/distiller/tests/wook')
    tflogger = TensorBoardLogger(msglogger.logdir)
    pylogger = PythonLogger(msglogger)
    loggers = [tflogger, pylogger]

    # train
    activations_collectors = image_classifier.create_activation_stats_collectors(model, args.activation_stats)
    for epoch in range(0, 200):
        with collectors_context(activations_collectors["train"]) as collectors:
            image_classifier.train(train_loader, model, criterion, optimizer, epoch=199, compression_scheduler=compression_scheduler, loggers=loggers, args=args)



if __name__ == '__main__':
    main()
