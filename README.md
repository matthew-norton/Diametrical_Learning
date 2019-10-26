# SGD-based Diametrical Risk Minimization (DRM) in Pytorch
This is an implementation of the SGD-DRM algorithm from the paper "Diametrical Risk Minimization: Theory and Computations", M. Norton and J. O. Royset. https://arxiv.org/abs/1910.10844. Experiments can be recreated by simply running the "run" scripts, depending on which experiment (network and dataset) you would like to reproduce. 


The code is complex because DRM is not straightforward to implement with construction of the Pytorch library. A network needs to be made from custom layers (found in the "layers" module) which allow for random "shifts" to the parameters of each layer during the forward operations. 


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
