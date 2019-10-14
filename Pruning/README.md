# STEP 2: Pruning of PyramidNet

The pruning method used is inspired of this [paper](https://arxiv.org/abs/1905.05934). It's based on the computation
of a Hessian matrix and the study of its eigenvalues in order to find the best way to prune. No sparsity is used during
this part of the process. Parameters are effectivly removed of the network.


## Run

### Prerequisite:
1. Download CIFAR-100 dataset if not already done.
2. Be sure to have the checkpoint of Step 1 put in *MicroNet/Training/trained_weights/pyramidnet*.
3. Build and Run the Docker Image *pruning*. **See instructions** in *MicroNet/Docker/myimages*.


### How to run:
```
cd /usr/share/bind_mount/scripts/MicroNet/Pruning/
CUDA_VISIBLE_DEVICES=0,1 python main_prune.py --config_path config.json
```

Please note that the commands is meant to be use with two GPUs. You can't use more with this implementation.
To only use one GPU, open *config.json* and replace the argument ***pruner_id: 1*** by ***pruner_id:0***


### Important note:
The pruning is done with respect to **specific batch sizes** chosen to optimize the use of my GPU ressources.
You may want to change them with respect to your limitations. To do so modify **line 80** of the file *main_prune.py*
with your values.