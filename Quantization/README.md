# Step 4: Quantization

The final part of the process consists in doing a uniform *fixed point* quantization of the network with a sharing
common exponents for parameters of the same layer.
All *weights* are quantized to ***9 bits*** (except for submission 2 in which case it's ***10 bits***).
All *activations* are quantized to ***12 bits***.

## Run

### Prerequisite:
1. Download CIFAR-100 dataset if not already done.
2. Be sure to have the checkpoint of Step 3 put in *MicroNet/Sparsity/sparse_weights/pyramidnet/pyramidnet_noskip_back*.
3. Build and Run the Docker Image *pruning* if not already done. **See instructions** in *MicroNet/Docker/myimages*.


### How to run:
```
cd /usr/share/bind_mount/scripts/MicroNet/Quantization
CUDA_VISIBLE_DEVICES=0 python main.py --config_path config.json
```

Please note that the commands is meant to be use with only one GPU.

### Important notes:

- You may want to change the batch size according to your ressources. To do so, change the *batch_size* argument in the
*config.json* file.

- The argument *checkpoint_file* allows you to choose the checkpoint of *step 2* you want to use. Please remember
that the checkpoint file format is ***checkpoint\_run\_[nb]\_[acc]\_[compression]\_[sparsity].pth***. Try to choose a
checkpoint being a good trade-off between compression and accuracy as you need a soft accuracy margin to need to go
through *step 4 (quantization)* and still being above the challenge accuracy threshold (***80%***).



