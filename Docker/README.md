# Docker Commands

## Build

### Fast AutoAugment (used for *step 1*)
```
cd Micronet/Docker/myimages/fast_autoaugment
docker build --no-cache --build-arg CUDA_VERSION=10.0 --build-arg CUDNN_VERSION=7 -t mhariat/fast_autoaugment .
```

*It may take some time...*

### Pruning (used for *step 1, 2, 3, 4, 5*)
```
cd Micronet/Docker/myimages/pruning
docker build --no-cache --build-arg CUDA_VERSION=10.0 --build-arg CUDNN_VERSION=7 -t mhariat/pruning .
```

## Run

### Fast AutoAugment
```
docker run -v /local/ml/mhariat/cifar_100:/usr/share/bind_mount/data/cifar_100 \
           -v /home/mhariat/MicroNet:/usr/share/bind_mount/scripts/MicroNet \
           --runtime nvidia \
           --name micronet \
           --net host \
           --ipc host \
           --init \
           --shm-size=2gb \
           -it mhariat/fast_autoaugment
```

### Pruning
```
docker run -v /local/ml/mhariat/cifar_100:/usr/share/bind_mount/data/cifar_100 \
           -v /home/mhariat/MicroNet:/usr/share/bind_mount/scripts/MicroNet \
           --runtime nvidia \
           --name micronet \
           --net host \
           --ipc host \
           --init \
           --shm-size=2gb \
           -it mhariat/pruning
```

### Important notes:

- The CIFAR-100 images are assumed to be at ***/local/ml/mhariat/cifar_100***. The MicroNet directory is assumed to be at
***/home/mhariat/MicroNet***. You may want to change the commands according to the location of your files/directories.

- The CIFAR-100 images should be put in a directory named ***cifar_100***.

- In the directory *cifar_100*, training images should be put in a sub-directory named ***training_set*** and the test
images in a sub-directory named ***test_set***.


Try to respect as much as possible the syntax to avoid errors.

Please respect the way scripts and data are bind mounted to the docker.
