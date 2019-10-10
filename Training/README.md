# Training of PyramidNet + ShakeDrop (+ SkipNet) [STEP 1]

For the Micronet Challenge we chose to start from PyramidNet and then to compress it. Multiple reasons led to choose
such a network. Amongst them:

- High accuracy (see table below)
- Reasonable number of parameters (26.2M) compared to Wide-Resnet-28-10 (36.5M)

The code is mainly a git clone of the Official [Fast AutoAugment](https://arxiv.org/abs/1905.00397) implementation in
PyTorch.

Fast Autoaugment is a simplified version of Autoaugment.

- Fast AutoAugment learns augmentation policies using a more efficient search strategy based on density matching.
- Fast AutoAugment speeds up the search time by orders of magnitude while maintaining the comparable performances.

One of the author (Ildoo Kim) kindly gave us the policies obtained with Shake-Shake(26_2x96d) on CIFAR-100.

We then added to PyramidNet a dynamic routing in convolutional network named "SkipNet". For more details on this method
please refer to the [paper](https://arxiv.org/pdf/1711.09485.pdf).
SkipNet learns to route images through a subset of layers on a per-input basis. Challenging images are routed through
more layers than easy images. We talk about two model designs with both feedforward gates and reccurent gates which
enable different levels of parameter sharing in the paper.

## Table Results of Fast-Autoaugment

### CIFAR-10 / 100

Search : **3.5 GPU Hours (1428x faster than AutoAugment)**, WResNet-40x2 on Reduced CIFAR-10

| Model(CIFAR-10)         | Baseline   | Cutout     | AutoAugment | Fast AutoAugment<br/>(transfer/direct) |
|-------------------------|------------|------------|-------------|------------------|
| Wide-ResNet-40-2        | 5.3        | 4.1        | 3.7         | 3.6 / 3.7        |
| Wide-ResNet-28-10       | 3.9        | 3.1        | 2.6         | 2.7 / 2.7        |
| Shake-Shake(26 2x32d)   | 3.6        | 3.0        | 2.5         | 2.7 / 2.5        |
| Shake-Shake(26 2x96d)   | 2.9        | 2.6        | 2.0         | 2.0 / 2.0        |
| Shake-Shake(26 2x112d)  | 2.8        | 2.6        | 1.9         | 2.0 / 1.9        |
| PyramidNet+ShakeDrop    | 2.7        | 2.3        | 1.5         | 1.8 / 1.7        |

| Model(CIFAR-100)      | Baseline   | Cutout     | AutoAugment | Fast AutoAugment<br/>(transfer/direct) |
|-----------------------|------------|------------|-------------|------------------|
| Wide-ResNet-40-2      | 26.0       | 25.2       | 20.7        | 20.6 / 20.6      |
| Wide-ResNet-28-10     | 18.8       | 28.4       | 17.1        | 17.8 / 17.5      |
| Shake-Shake(26 2x96d) | 17.1       | 16.0       | 14.3        | 14.9 / 14.6      |
| PyramidNet+ShakeDrop  | 14.0       | 12.2       | 10.7        | 11.9 / 11.7      |

### ImageNet

Search : **450 GPU Hours (33x faster than AutoAugment)**, ResNet-50 on Reduced ImageNet

| Model      | Baseline   | AutoAugment | Fast AutoAugment |
|------------|------------|-------------|------------------|
| ResNet-50  | 23.7 / 6.9 | 22.4 / 6.2  | **22.4 / 6.3**   |
| ResNet-200 | 21.5 / 5.8 | 20.0 / 5.0  | **19.4 / 4.7**   |


## Run

Please follow these steps to Launch the training process:

- First, build the Docker Image. To do so go to "~/MicroNet/Docker/myimages/fast_autoaugment" and run:

docker build --no-cache --build-arg CUDA_VERSION=10.0 --build-arg CUDNN_VERSION=7 -t mhariat/fast_autoaugment .

It may take some time...

- Then run the docker with the following command:

docker run -v /local/ml/mhariat/cifar_100:/usr/share/bind_mount/data/cifar_100 \
           -v /home/mhariat/MicroNet:/usr/share/bind_mount/scripts/MicroNet \
           --runtime nvidia \
           --name micronet \
           --net host \
           --ipc host \
           --init \
           --shm-size=2gb \
           -it mhariat/fast_autoaugment

The cifar_100 images are assumed to be at "/local/ml/mhariat/cifar_100". In the directory "cifar_100", training images
should be put in a sub-directory named "training_set" and the test images in a sub-directory named "test_set".

The Micronet directory is assumed to be at "/home/mhariat/Micronet"

You may have to change the command according to the location of your files/directories. However try to respect as much
as possible the syntax to avoid errors.

- Once in the docker, go to : "/usr/share/bind_mount/scripts/MicroNet/Training" and run the two following commands:

1. pip install -r requirements.txt

2. export PYTHONPATH="$PYTHONPATH:/usr/share/bind_mount/scripts/MicroNet/Training"

- Finally go to "/usr/share/bind_mount/scripts/MicroNet/Training/FastAutoAugment" and launch the training with
the command:

horovodrun -np 4 --cache-capacity 2048 python train.py -c confs/pyramid272a200b.yaml --aug fa_shake26_2x96d_cifar100 --dataset cifar100 --batch 64 --dataroot /usr/share/bind_mount/data/cifar_100 --tag pyramidskipnet_bottleneck_fa_shake26_2x96d_cifar100 --save /app/results/checkpoints  --horovod

You may have to change the number of GPU (here 4: "-np 4") and the batch size (here 64: "--batch 64") according to your
computational ressources.

To train PyramidNet + SkipNet go to confs/pyramid272a200b.yaml and change "pyramid" by "pyramid_skip" in model/type.

The checkpoint weights are regularly (every 10 epochs) put in the directory "/app/results/checkpoints"

