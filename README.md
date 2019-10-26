# SGD-based Diametrical Risk Minimization (DRM) in Pytorch
This is an implementation of the SGD-DRM algorithm from the paper "Diametrical Risk Minimization: Theory and Computations", M. Norton and J. O. Royset. https://arxiv.org/abs/1910.10844. 

### How to run? 
After cloning the repository, the experiments from the paper can be recreated by simply going to the command line and running:

python run_(dataset)_(network).py path_to_put_downloaded_data

where "run_(dataset)_(network).py" is one of the provided "run" files and "path_to_put_downloaded_data" is somewhere pytorch can download and put CIFAR10 and/or MNIST. The output will be two plots, one providing the progress of test accuracy during training and the other plotting the approximate distribution of neighborhood losses around the optimal solutions. 

The code is set up to train two networks; one trained with DRM and the other with ERM. The "run" files walk through the necissary components of running a model. However, parts of the code can be easily customized to experiment with other architectures and variations of the training routine. 



Requirements:

- Python 3.6+
- pytorch 1.2+
- torchvision 0.4+
- numpy 1.17+
- matplotlib 3.1.1+ (for utils.plot_net )
- seaborn 0.9+ (for diametric.sample_neighborhood_losses)
- pandas 0.25+ (for diametric.sample_neighborhood_losses)


Organization of Modules:

models
  - Simple_Net
  - resnet20 
  - DRM_net

layers
  - DRM compatible Linear and Conv2d layers


fast_random
  - MultithreadedRNG class for fast parallel sampling of random directions.


utils
  - save_nets
  - load_nets
  - plot_net

datastore
  - Data loading classes and functions


diametrical
  - build_net 
  - assess
  - train
  - test
  - sample_neighborhood
