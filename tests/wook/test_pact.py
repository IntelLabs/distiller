import distiller
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from distiller.models import create_model


def main():
    yaml_path = '../../examples/quantization/quant_aware_train/preact_resnet20_cifar_pact.yaml'
    model = create_model(False, 'cifar10', 'preact_resnet20_cifar',
                         parallel=not False, device_ids=None)
    print(model)

    compression_scheduler = None
    optimizer = None
    start_epoch = 0

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=0.0002)

    compression_scheduler = distiller.file_config(model, optimizer, yaml_path, compression_scheduler, None)

    print(model)


if __name__ == '__main__':
    main()