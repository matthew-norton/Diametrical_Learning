# SGD-based Diametrical Risk Minimization (DRM) in Pytorch
This is an implementation of the SGD-DRM algorithm from paper [1] : "Diametrical Risk Minimization: Theory and Computations", M. Norton and J. O. Royset. https://arxiv.org/abs/1910.10844.

Requirements:

- Python 3.6+
- pytorch 1.2+
- torchvision 0.4+
- numpy 1.17+
- matplotlib 3.1.1+ (for utils.plot_net )
- seaborn 0.9+ (for drm_train_test.sample_neighborhood_losses)
- pandas 0.25+ (for drm_train_test.sample_neighborhood_losses)

### Quick Start: How to recreate experiments from [1]?
After cloning this repository, you can recreate experiments from paper [1] by simply executing one of the scripts "run_CIFAR10_FC.py" , "run_CIFAR10_resnet20.py", or "run_MNIST_FC.py" The scripts can be run as-is via the command line and only require a single command line argument as follows:

`python run_(dataset)_(network).py path_to_put_downloaded_data`

where "path_to_put_downloaded_data" is somewhere pytorch can download and put the CIFAR10 and/or MNIST datasets. The output will be two plots, one providing the progress of test accuracy during training and the other plotting the approximate distribution of neighborhood losses around the optimal solutions. NOTE: Execution can take quite a while and one should utilize the options in the script for saving intermediate network parameters and performance statistics.


## Implementation details:
To train your own custom network via SGD-DRM, the steps below provide a quick guide. Full examples can be found in the "run" files mentioned above. NOTE: The current implementation is set up to perform classification with NLL loss function. Changes to the loss function and network output should be addressed in the Wrapper_Net.regular_forward method (in the models.wrap_net module), as well as the train and test functions found in the drm_train_test module.

1)  Create your network as a standard pytorch object (see pytorch tutorials). We will call this the "core_network."  

`my_core_network = My_Net().to(device)`

Note that all parameters need to be named. This will happen automatically if you follow standard pytorch conventions. You can check this by printing all of your named parameters: `print([name for name in my_core_network.named_parameters()])`

2)  Wrap the "core_network" in a `Wrapper_Net`

`new_net = Wrapper_Net(core_network=my_core_network)`

Along with some useful performance tracking attributes, `Wrapper_Net` objects make the network compatible with SGD-DRM. First, when initialized, the object initializes multithreaded random number generators for each layer of the network. This helps for fast sampling during SGD-DRM. Second, it has an `assess` function which calculates the loss value for `num_dir` random points in a `gamma` neighborhood. Third, it has a special forward function to calculate the maximum loss over the finite set V (see paper).

3) Create an optimizer to keep track of gradients

`my_optimizer = optim.SGD(new_net.parameters() ,momentum=0 ,lr=.01)`

4) Use the train and test methods from `drm_train_test` module.

`sampling_interval = 5`

`num_dir=20`

`g = 1`

`epoch = 1`

`#This trains for one epoch.`

`DRM_SGD_train(epoch,new_net,my_optimizer,train_loader,num_dir=num_dir, gamma=g,sampling_interval=sampling_interval)`
