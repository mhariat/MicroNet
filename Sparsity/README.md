# Step 3: Sparsity

This part of the process consists in adding sparsity to the network. It's a simple element-wise pruning method
based on the magnitude of the parameters.


## Run

### Prerequisite:
1. Download CIFAR-100 dataset if not already done.
2. Be sure to have the checkpoint of Step 2 put in *MicroNet/Pruning/pruned_weights/pyramidnet/pyramidnet_noskip_back*.
3. Build and Run the Docker Image *pruning* if not already done. **See instructions** in *MicroNet/Docker/myimages*.


### How to run:
```
cd /usr/share/bind_mount/scripts/MicroNet/Sparsity
CUDA_VISIBLE_DEVICES=0 python main.py --config_path config.json
```

Please note that the commands is meant to be use with only one GPU.

### Important notes:

- You may want to change the batch size according to your resources. To do so, change the *batch_size* argument in the
*config.json* file.

- The argument *checkpoint_file* allows you to choose the checkpoint of *step 2* you want to use. Please remember
that the checkpoint file format is ***checkpoint\_run\_[nb]\_[acc]\_[compression].pth***. Try to choose a checkpoint
being a good trade-off between compression and accuracy as you need a soft accuracy margin to need to go through
*step 3 (sparsity)* and *step 4 (quantization)* and still being above the challenge accuracy threshold (***80%***).




