## Lottery Ticket Hypothesis

>The Lottery Ticket Hypothesis: A randomly-initialized, dense neural network contains a subnetwork that is initialized
such that — when trained in isolation — it can match the test accuracy of the original network after training for at
most the same number of iterations."

### Finding winning tickets
> We identify winning tickets by training networks and subsequently pruning
their smallest-magnitude weights. The set of connections that survives this process is the architecture
of a winning ticket. Unique to our work, the winning ticket’s weights are the values to which these
connections were initialized before training. This forms our central experiment:
>1. Randomly initialize a neural network f(x; theta_0) (where theta_0 ~ D_0).
>2. Train the network for j iterations, reaching parameters theta_j.
>3. Prune s% of the parameters, creating a mask m where Pm = (100 - s)%.
>4. To extract the winning ticket, reset the remaining parameters to their values in theta_0, creating
the untrained network f(x; m * theta_0).

### Example
Train a ResNet20-CIFAR10 network from scratch, and save the untrained, randomized initial network weights in a checkpoint file.
To do this, you use the `--save-untrained-model` flag: <br>
```bash
python3 compress_classifier.py --arch resnet20_cifar  ${CIFAR10_PATH} -p=50 --epochs=110 --compress=../ssl/resnet20_cifar_baseline_training.yaml --vs=0 --gpus=0 -j=4 --lr=0.4 --name=resnet20 --save-untrained-model
``` 

After training the network, we have two outputs: the best trained network (`resnet20_best.pth.tar`) and the initial untrained network (`resnet20_untrained_checkpoint.pth.tar`).<br>
In this example, we copy both checkpoints into the `examples/lottery_ticket` directory for convenience.

```bash
cp logs/resnet20___2019.08.22-220243/resnet20_best.pth.tar ../lottery_ticket/
cp logs/resnet20___2019.08.22-220243/resnet20_untrained_checkpoint.pth.tar ../lottery_ticket/
```

We then prune our best trained ResNet20 network and copy the result into `examples/lottery_ticket` as well.
```bash
python3 compress_classifier.py --arch resnet20_cifar  ../../../data.cifar10 -p=50 --lr=0.3 --epochs=180 --compress=../agp-pruning/resnet20_filters.schedule_agp_4.yaml  --resume-from=../lottery_ticket/resnet20_best.pth.tar --vs=0 --reset-optimizer --gpus=0
cp logs/2019.08.22-222752/best.pth.tar ../lottery_ticket/resnet20_pruned.pth.tar
```

Next, we run ```lottery.py``` to extract the winning ticket.
```bash
python lottery.py --lt-untrained-ckpt=resnet20_untrained_checkpoint.pth.tar --lt-pruned-ckpt=resnet20_pruned.pth.tar
```

Finally, we train the winning ticket.
```bash
python3 compress_classifier.py --arch resnet20_cifar  ../../../data.cifar10 -p=50 --lr=0.1 --epochs=180  --resume-from=../lottery_ticket/resnet20_untrained_checkpoint.pth.tar_lottery_checkpoint.pth.tar --vs=0 --reset-optimizer --gpus=0 --compress=../ssl/resnet20_cifar_baseline_training.yaml
```


[1] Jonathan Frankle, Michael Carbin<br>
    The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks<br>
    arXiv:1803.03635