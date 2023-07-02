# Paper replications for Deep Double Descent

## Belkin et al. 2018

I replicate the experiment showcasing double descent in a Fully Connected Network trained on MNIST as described in the paper. The notebook MNIST_FCNN.ipynb shows a prototype of the workflow this repository follows.

## Nakkiran et al. 2019 (In progress)

This repository replicates the experiment from Nakkiran et al. 2019 that shows double descent in Resnet models when trained on the CIFAR-100 dataset and can be executed using:

`make resnet`

The make command will prompt you to input a value of k, where k can range from 1 to 64 according to the original experiment. For every k, there are logs dumped that can be used to plot the double descent curve. 
