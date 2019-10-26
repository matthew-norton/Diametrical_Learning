from layers import Conv2d_Shift , Linear_Shift
from fast_random import MultithreadedRNG

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#############################################################################################
'''Assess loss over batch "x" for
"num_dir" random points within a "gamma"
neighborhood of the current point'''
#############################################################################################

def assess(net,x ,target, seed=0, num_dir = 10 , gamma = 2 , return_v = False , return_all_losses=False):
    losses = []
    pert_v = {}

    rng_dict = net.rng_dict

    with torch.no_grad():
        np.random.seed(seed)
        for k_ in range(num_dir):
            for name,module in net.named_modules():
                if isinstance(module , (Linear_Shift, Conv2d_Shift)):
                    pert_v[name] = {'weight' : 0 ,
                                  'bias' : 0 }

                    pert_v[name]['weight'] = rng_dict[name]['weight'].fill().to(net.device)

                    pert_v[name]['weight'] *= gamma /torch.norm(pert_v[name]['weight'])

                    module.weight_shift = pert_v[name]['weight']

                    pert_v[name]['bias'] = torch.tensor(0,device=net.device)
                    if module.bias is not None:

                        pert_v[name]['bias'] = rng_dict[name]['bias'].fill().to(net.device)

                        pert_v[name]['bias'] *= gamma/torch.norm(pert_v[name]['bias'])

                        module.bias_shift = pert_v[name]['bias']

            loss = net(x , target = target , train = False , return_probs=False)
            losses.append([ loss , pert_v ])


        for name,module in net.named_modules():
            if name in pert_v.keys():
                module.weight_shift = torch.tensor(0,device=net.device)
                module.bias_shift = torch.tensor(0,device=net.device)


        losses = np.array(losses)

        if return_all_losses:
            return losses[:,0]

        elif return_v:
            idx = np.argmax(np.array(losses)[:,0])
            return losses[idx,1]

        else:
            return losses
#############################################################################################
'''TRAIN AND TEST ROUTINES'''
#############################################################################################


import time
def DRM_SGD_train(epoch,
       network,
       optimizer,
       train_loader,
       gamma = 0,
       log_interval = 10,
       sampling_interval = 5
      ):
    network.train()
    np.random.seed(0)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        data = data.to(network.device)
        target = target.to(network.device)

        if batch_idx%sampling_interval==0:
            #s=time.time()
            v_new = assess(network,data ,target, seed=epoch*batch_idx, num_dir = network.num_dir , gamma = gamma , return_v = True)
            network.v.append(v_new)
            #print('assess time: ' , time.time() - s)

        #s = time.time()
        loss = network(data,target=target)

        loss.backward()
        optimizer.step()
        #print('loss time DRM: ' , time.time() - s)

        network.train_loss.append(loss.item())
        network.train_steps+=1

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))





def ERM_SGD_train(epoch,network,optimizer,train_loader,log_interval=10):
    network.train()
    np.random.seed(0)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()


        data = data.to(network.device)
        target = target.to(network.device)

        #s = time.time()
        output = network(data , return_probs=True)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        #print('loss time ERM: ' , time.time() - s)

        network.train_loss.append(loss.item())
        network.train_steps+=1

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))




def test(network,test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(network.device)
            target = target.to(network.device)

            output = network(data , return_probs = True , train = False)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)

        network.test_loss.append([test_loss, network.train_steps])

        network.test_acc.append([ 100. * correct / len(test_loader.dataset), network.train_steps])
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




def get_train_acc(network,train_loader):
    network.eval()
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(network.device)
            target = target.to(network.device)

            output = network(data , return_probs = True , train = False)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

        network.train_acc.append([ 100. * correct / len(train_loader.dataset), network.train_steps])
        print('\nTrain set Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))




#############################################################################################
'''GET FINAL ASSESSMENT LOSSES AND FINAL BASE LOSS OF TRAINED CLASSIFIERS'''
#############################################################################################

def sample_neighborhood_losses(network1,network2,
                                train_loader, save_path = None,
                                num_dir = 100 , gamma = 10):

    assess_losses={'network1': [] ,
                   'network2': [] }
    base_loss = {'network1': 0 ,
                   'network2': 0 }




    for net , net_name in zip([network1,network2],['network1','network2']):



        for batch_idx , (data , target) in enumerate(train_loader):
            with torch.no_grad():
                data = data.to(network1.device)
                target = target.to(network1.device)


                network1.rng_dict = { name :  { 'weight' : MultithreadedRNG(module.weight.shape, seed=0 , threads = 8) ,
                      'bias' : MultithreadedRNG(module.bias.shape, seed=0 , threads = 2) if module.bias is not None else 0 }
                    for name , module in network1.named_modules() if isinstance(module , (Linear_Shift, Conv2d_Shift)) }

                network2.rng_dict = { name :  { 'weight' : MultithreadedRNG(module.weight.shape, seed=0 , threads = 8) ,
                                  'bias' : MultithreadedRNG(module.bias.shape, seed=0 , threads = 2) if module.bias is not None else 0 }
                    for name , module in network2.named_modules() if isinstance(module , (Linear_Shift, Conv2d_Shift)) }



                losses = assess(net,data,target , num_dir = num_dir , gamma = gamma, return_v=False, return_all_losses=True)

                if batch_idx==0:
                    base_loss[net_name] =  net(data,target=target,train=False,return_probs=False)*(len(target)/len(train_loader.dataset))
                    assess_losses[net_name] = losses*(len(target)/len(train_loader.dataset))
                else:
                    base_loss[net_name] +=  net(data,target=target,train=False,return_probs=False)*(len(target)/len(train_loader.dataset))
                    assess_losses[net_name] += losses*(len(target)/len(train_loader.dataset))

                print('Train base: {} [{}/{} ({:.0f}%)]\t max Loss: {:.6f}'.format(
                    base_loss[net_name], batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), assess_losses[net_name].max()))


    data_temp = []
    for key in ['network1','network2']:
        for i in assess_losses[key]:
            data_temp.append(  [ key , i.data.item() , base_loss[key].data.item() ])
    data_temp = pd.DataFrame( data = data_temp , columns = ['net_name','losses','base_loss'])


    grouped = data_temp.groupby('net_name')['losses']
    fig,(ax,ax2) = plt.subplots(1, 2, sharey=False)
    fig.set_size_inches(4.5*2, 2.25*2)
    fig.dpi=100
    # plot the same data on both axes
    sns.distplot(grouped.get_group('network1') ,
                 kde = False ,
                 bins = 40 ,
                 ax = ax ,
                 color = 'blue')

    ax.axvline( base_loss['network1'].data.item() ,
               color='blue' ,
               label = 'DRM Solution Loss',
               ymin = 0 ,
               ymax = 1 ,
               linestyle ='dashed')
    ax.set_title('Diametric Neighborhood Losses')
    ax.set_xlabel('Train Loss')
    ax.set_ylabel('Frequency')
    ax.legend()
    sns.distplot(grouped.get_group('network2') ,
                 kde = False ,
                 bins = 40 ,
                 ax = ax2,
                 color = 'red')

    ax2.axvline( base_loss['network2'].data.item() ,
                color='red' ,
                label = 'ERM Solution Loss',
                ymin = 0 ,
                ymax = 1 ,
                linestyle ='dashed')
    ax2.set_title('ERM Neighborhood Losses')
    ax2.set_xlabel('Train Loss')
    ax2.legend()

    if save_path is None:
        plt.savefig('temp_plot_neighborhood_losses.pdf' , dpi=600)
    else:
        plt.savefig(save_path , dpi=600)

    return data_temp
