# <p align="center"> MicroNet Challenge - NeurIPS 2019 </p>

This repository is our solution to the MicroNet challenge hosted at ***NeurIPS 2019***.

## How to Test

To obtain the final MicroNet score go to *MicroNet/Test* and execute *main.py*. Please ***read carefully the README
file before to know exactly how to do so***.

You may also be interested in taking a look at the short report in *Micronet/Reports*.

## Results

| Submission        | Accuracy   | Score    |
|-------------------------|------------|------------|
| 1        | 80.11%        | 0.0467        |
| 2       | 80.47%       | 0.0464        |
| 3   | 80.003%        | 0.0456        |


## Team

- Marwane Hariat, marwane.hariat@gmail.com, Intern at Wave Computing, MS Student at ENS Paris-Saclay and École des Ponts ParisTech.

- Sikandar Mashayak, symashayak@gmail.com, Wave Computing.

- Sylvain Flamant, sflamant@gmail.com, Wave Computing.



## Description

We addressed the challenge by splitting the work into 4 different steps.

- Step 1: Training

We decided to start from **[PyramidNet]**(https://arxiv.org/pdf/1610.02915.pdf) (bottleneck, depth: 270, &alpha;=200)
trained using **[Fast-AutoAugment]**(https://arxiv.org/abs/1905.00397). We achieve *88.3% of accuracy*.

The data augmentation **policy** obtained with **Shake-Shake(26_2x96d)** using **only CIFAR-100** was given by the
authors.

Please see folder *Micronet/Training* and the corresponding README file for more details on this part of the process.

- Step 2: Pruning

The pruning method used in this part is inspired of this [paper](https://arxiv.org/abs/1905.05934).
It's based on the computation of a Hessian matrix and the study of its eigenvalues in order to find the best way to
prune. No sparsity is used during this part of the process. Parameters are effectively removed of the network.

At the end of this part we are able to get a network effectively *compressed at 92.81%* with *81.30% of accuracy*.

Please see folder *Micronet/Pruning* and the corresponding README file for more details on this part of the process.

- Step 3: Sparsity

This part of the process is a simple element-wise pruning method based on the magnitude of the parameters. It aims at
adding sparsity to gain a little more compression. We were able to add *45.31% of sparsity* in the already compressed
network.

Please see folder *Micronet/Sparsity* and the corresponding README file for more details on this part of the process.


- Step 4: Quantization

The final part of the process consists in doing a uniform *fixed point* quantization of the network with a sharing
common exponents for parameters of the same layer.

All *weights* are quantized to ***9 bits*** (except for submission 2 in which case it's ***10 bits***).

All *activations* are quantized to ***12 bits***.

Please see folder *Micronet/Quantization* and the corresponding README file for more details on this part of the process.


- Step 5: Test

When all of the previous steps are done, we put the checkpoint weights in *Micronet/FinalWeights*.

One may notice a checkpoint named *checkpoint_sparsity.pth*. It's the checkpoint obtained at the end of *step 3*
and common to all submissions.
It's used to create the compressed network (much more different than the original PyramidNet as we truly remove
parameters during *step 2*).
Then, weights of *step 4*, stored in files named *checkpoint_quantization\_[n]\_submission.pth, are loaded.

Please see folder *Micronet/Test* and the corresponding README file for more details on this part of the process.


