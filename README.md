# <p align="center"> MicroNet Challenge - NeurIPS 2019 </p>

This repository is our solution to the MicroNet challenge hosted at ***NeurIPS 2019***.

## Testing

To obtain the final MicroNet score go to *MicroNet/Test* and execute *main.py*. Please **read carefully the
corresponding** ***README*** **file before, to know exactly what to do**.

You may also be interested in taking a look at the short **report** in *Micronet/Reports*.

## Results with quantization

| Submission| Total Parameters| Total Flops| Accuracy| Score|
|-----------|------------|------------|------------|------------|
| 1         | 0.3445M     | 391.2M     | 80.11%     | 0.0467     |
| 2         | 0.3558M    | 384.8M     | 80.46%     | 0.0464     |
| 3         | 0.32467M    | 384.8M | 80.01%     | 0.04558    |


## Results without quantization

| Submission| Total Parameters (w/o Sparsity)| Total Parameters (w/ Sparsity)| Total Flops (w/ Sparsity)| Accuracy| Score|
|-----------|------------|------------|------------|------------|------------|
| 1         | 1.89M    | 1.011M     | 580M       | 80.11%     | 0.0467     |
| 2         | 1.826M    | 0.951M     | 560M       | 80.46%     | 0.0464     |
| 3         | 1.826M    | 0.951M     | 560M       | 80.01%     | 0.04558    |


## Team

- **Marwane Hariat**, marwane.hariat@gmail.com, Intern at *Wave Computing*, MS Student at *ENS Paris-Saclay* and *École des Ponts ParisTech*.

- **Sikandar Mashayak**, symashayak@gmail.com, *Wave Computing*.

- **Sylvain Flamant**, sflamant@gmail.com, *Wave Computing*.



## Description

We addressed the challenge by splitting the work into 4 different steps.

- **Step 1: Training**

We decided to start from [PyramidNet](https://arxiv.org/pdf/1610.02915.pdf) trained using
[Fast-AutoAugment](https://arxiv.org/abs/1905.00397). We achieved **88% of accuracy**.

Please see folder *Micronet/Training* and the corresponding *README* file for more details on this part of the process.

- **Step 2: Pruning**

The pruning method used in this part is inspired of this [paper](https://arxiv.org/abs/1905.05934) and based on the
computation of a **hessian matrix** and the study of its **eigenvalues** in order to find the best way to prune.

The pruning is done gradually with regular **fine-tuning** and **annealing** in between two pruning stages.
The annealing phase consists in slightly increasing the size of the network artificially.

No sparsity is used during this step. Parameters are **effectively removed** of the network.

At the end of this part we are able to get a network effectively **compressed at 92.81%** with **81.30% of accuracy**.

Please see folder *Micronet/Pruning* and the corresponding README file for more details on this part of the process.

- **Step 3: Sparsity**

This part of the process is a simple element-wise pruning method based on the **magnitude** of the parameters. We were
able to add **45.31% of sparsity** to the already very compressed network.

Please see folder *Micronet/Sparsity* and the corresponding *README* file for more details on this part of the process.


- **Step 4: Quantization**

The final part of the process consists in doing a uniform **fixed point** quantization of the network with a sharing
common exponents for parameters of the same layer.

All *weights* are quantized to ***9 bits*** (except for submission 2 in which case it's ***10 bits***).
All *activations* are quantized to ***12 bits***.

Please see folder *Micronet/Quantization* and the corresponding *README* file for more details on this part of the process.


- **Step 5: Test**

When all of the previous steps are done, checkpoint are put in *Micronet/FinalWeights*.

One may notice a checkpoint named *checkpoint_sparsity.pth* coming from *step 3* and common to all submissions.
It's used to create the compressed network (much more different than the original PyramidNet as we truly remove
parameters during *step 2*).

Then, checkpoints of *step 4*, stored in files named *checkpoint_quantization\_[n]\_submission.pth*, are loaded.

Please see folder *Micronet/Test* and the corresponding *README* file for more details on this part of the process.


