# Step 5: Test


## Run

### Prerequisite:
1. Download CIFAR-100 dataset if not already done.
2. Be sure to have the checkpoint of *step 4* and *step 5* put in *MicroNet/FinalWeights*.
3. Build and Run the Docker Image *pruning* if not already done. **See instructions** in *MicroNet/Docker/myimages*.


### How to run:
```
cd /usr/share/bind_mount/scripts/MicroNet/Test
CUDA_VISIBLE_DEVICES=0 python main.py --config_path config.json
```

Please note that the commands is meant to be use with only one GPU.

### Important notes:

- You may want to change the batch size according to your ressources. To do so, change the *batch_size* argument in the
*config.json* file.

- The argument *submission* allows you to choose the submission you want to use. Please note that **submission 2** is
done with a weight quantization of *10 bits*. And require you to change the argument *param_bits* in the
*config.json* file from ***param_bits: 9*** to ***param_bits: 10***


### Final Scores

| Submission        | Accuracy   | Score    |
|-------------------------|------------|------------|
| 1        | 80.11%        | 0.0467        |
| 2       | 80.46%       | 0.0464        |
| 3   | 80.01%        | 0.04558        |



