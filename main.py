#############################################################################################
'''SET PARAMETERS'''
#############################################################################################
save_freq = 10 #save network parameters every save_freq epochs
save_now = False #should we save results for this run?
fname = None #file to save to if saving results

batch_size_train = 15000
batch_size_test = 1000
percent_flip = .5 #number of labels to flip


torch_random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(torch_random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#############################################################################################
'''GET DATA'''
#############################################################################################

from datastore import get_CIFAR10
path = '/home/matthewnorton/Documents/pytorch_data/'
exclude_list = [i for i in range(3,10)]
num_classes = 3
input_shape = (3,32,32)
train_loader, test_loader = get_CIFAR10(path,
                                        exclude_list,
                                        batch_size_train,
                                        batch_size_test,
                                        percent_flip)


#############################################################################################
'''BUILD CLASSIFIERS'''
#############################################################################################
from DRM_net import DRM_Net
from models.fc_net import Simple_Net
from models.resnet import resnet20

#resnet20(num_classes=num_classes) #
core_network1 = Simple_Net(num_classes=num_classes ,input_shape=input_shape)
drm_network = DRM_Net(core_network=core_network1,  ,sampling_workers=4 , num_dir=20 , device = device).to(device)

core_network2 = Simple_Net(num_classes=num_classes ,input_shape=input_shape)
erm_network = DRM_Net(core_network=core_network2 ,sampling_workers=2, device = device).to(device)


#############################################################################################
'''TRAIN CLASSIFIERS'''
#############################################################################################
import time
from utils import save_nets , plot_net
from diametrical import DRM_SGD_train , SGD_train , test

sampling_interval = 5
num_epochs =100
lr_schedule = dict([ (0,.01) , (100,.001)] )
gamma_schedule = dict( [(0,10) ,(100,10) ] )

test(drm_network,test_loader)
test(erm_network,test_loader)

for epoch in range(num_epochs):
    if epoch in lr_schedule.keys():
        optimizer_drm = optim.SGD(drm_network.parameters() ,momentum=0 ,lr=lr_schedule[epoch])
        optimizer_erm = optim.SGD(erm_network.parameters() ,momentum=0 ,lr=lr_schedule[epoch])
    if epoch in gamma_schedule.keys():
        g = gamma_schedule[epoch]

    s = time.time()
    DRM_SGD_train(epoch,drm_network,optimizer_drm,train_loader,
                    gamma=g , sampling_interval=sampling_interval)
    print(time.time() - s ,'anti time')

    s = time.time()
    SGD_train(epoch,erm_network,optimizer_erm,train_loader)
    print(time.time() - s , 'normal time')

    test(drm_network,test_loader)
    test(erm_network,test_loader)

    if epoch%save_freq==0:
        if save_now:
            save_nets(fname,drm_network,erm_network,
                            optimizer_drm,optimizer_erm,
                            lr_schedule,gamma_schedule)


plot_net(drm_network)
sample_neighborhood_losses(drm_network,erm_network,num_dir = 100 , gamma = 10)
