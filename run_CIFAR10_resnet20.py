
import torch

#############################################################################################
'''SET PARAMETERS'''
#############################################################################################
save_freq = 10 #save network parameters every save_freq epochs
save_now = False #should we save results for this run?
fname = None #file to save to if saving results



torch_random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(torch_random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#############################################################################################
'''GET DATA'''
#############################################################################################

from datastore import get_CIFAR10
import sys

path = sys.argv[1]# example --> path = '/home/user/pytorch_data/'
assert path is not None , "Need path to existing data or location to put downloaded data"

batch_size_train = 100 #batch size for training
batch_size_test = 1000 #batch size for testing
percent_flip = .5 #number of labels to flip

exclude_list = [i for i in range(3,10)] #we exclude 7 of 10 classes from the dataset.
num_classes = 3 #is three since we excluded 7 from 10 with exclude list
input_shape = (3,32,32) #need shape of the input data for later (when making network, need this for size of first layer in fc_net.Simple_Net)
train_loader, test_loader = get_CIFAR10(path,
                                        exclude_list,
                                        batch_size_train,
                                        batch_size_test,
                                        percent_flip)


#############################################################################################
'''BUILD CLASSIFIERS'''
#############################################################################################
'''We wrap the "core network" inside of a DRM_Net object. This object does a few things:
1) It allows the network to be trained with Diametrical Risk Minimization (i.e. the DRM_SGD_train function).
2) It has a lot of attributes that keep track of the networks performance stats as training progresses.
So even if you don't train with DRM_SGD_train, it can be useful to wrap it in this to track stats.

Note: When specifying the DRM_Net, we also specify the required params for performning DRM:
sampling_workers: number of threads used to perform sampling of random directions in DRM algorithm.
num_dir: number of random directions to sample. (i.e. size of set U in SGD-DRM algorithm in paper). Frequency of sampling (how often) is given by parameter "sampling_interval" which is specified below.
v_max_len: number of past vectors to keep in set V (see algorithm in paper)
lr_schedule: if we want to change the learning rate (i.e. step size for gradient steps) over the course of the algorithm
gamma_schedule: "gamma" used for sampling set U. Allows to change the size of the neighborhood over which we sample num_dir points over the course of the algorithm
'''
from models.DRM_net import DRM_Net
from models.fc_net import Simple_Net
from models.resnet import resnet20

num_dir=20
v_max_len = 1
sampling_interval = 5
num_epochs =801
lr_schedule = dict([ (0,.01) , (750,.001)] )
gamma_schedule = dict( [(0,1) ] )

core_network1 = resnet20(num_classes=num_classes)
drm_network = DRM_Net(core_network=core_network1 , v_max_len = v_max_len ,sampling_workers=4 , num_dir=num_dir , device = device).to(device)

core_network2 = resnet20(num_classes=num_classes)
erm_network = DRM_Net(core_network=core_network2 ,sampling_workers=2, device = device).to(device)


#############################################################################################
'''TRAIN CLASSIFIERS'''
#############################################################################################
from utils import save_nets , plot_net
from diametrical import DRM_SGD_train , ERM_SGD_train , test , sample_neighborhood_losses
import torch.optim as optim


test(drm_network,test_loader)
test(erm_network,test_loader)

for epoch in range(num_epochs):

    # check the schedule to see if we want to change our learning rate
    if epoch in lr_schedule.keys():

        optimizer_drm = optim.SGD(drm_network.parameters() ,momentum=0 ,lr=lr_schedule[epoch])
        optimizer_erm = optim.SGD(erm_network.parameters() ,momentum=0 ,lr=lr_schedule[epoch])

    # check the schedule to see if we want to change our neighborhood sampling radius
    if epoch in gamma_schedule.keys():

        g = gamma_schedule[epoch]


    #For the first network, we train using DRM (i.e. using the DRM_SGD_train function)
    DRM_SGD_train(epoch,drm_network,optimizer_drm,train_loader,
                    gamma=g , sampling_interval=sampling_interval)

    #For the second network, we train using regular ERM objective with SGD with the SGD_train function
    ERM_SGD_train(epoch,erm_network,optimizer_erm,train_loader)

    #check test set performance
    test(drm_network,test_loader)
    test(erm_network,test_loader)

    #plot currect history of performance for both networks
    plot_net([(drm_network,'DRM') , (erm_network,'ERM') ] )

    #save the network parameters if desired
    if epoch%save_freq==0:
        if save_now:
            save_nets(fname,drm_network,erm_network,
                            optimizer_drm,optimizer_erm,
                            lr_schedule,gamma_schedule)

#############################################################################################
'''PLOT CLASSIFIER TEST ACCURACY'''
#############################################################################################

plot_net([(drm_network,'DRM') , (erm_network,'ERM') ] )

#############################################################################################
'''Sample points in neighborhood of optimal solution and plot the distribution of losses'''
#############################################################################################


sample_neighborhood_losses(drm_network,erm_network,train_loader, num_dir = 10000 , gamma = 1)
