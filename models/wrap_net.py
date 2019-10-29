#############################################################################################
'''DEFINE CLASSIFIERS'''
#############################################################################################



from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_random import MultithreadedRNG

class Wrapper_Net(nn.Module):
    def __init__(self  ,
                 v_max_len = 1 ,
                sampling_workers = 8,
                sampling_seed = 0 ,
                core_network = None,
                device = None ,
                exclude_re =[]):
        super(Wrapper_Net, self).__init__()

        assert core_network is not None , "Need to specify a core_network. Try Simple_Net(num_classes=num_classes ,input_shape=input_shape)."

        self.core_network = core_network
        self.device = device
        self.v_max_len = v_max_len
        self.v = deque(maxlen=v_max_len)

        self.train_loss = []
        self.test_loss =[]
        self.train_acc=[]
        self.test_acc =[]
        self.train_steps =0


        self.exclude_params = []
        for name , param in self.named_parameters():
            for pattern in exclude_re:
                 if pattern in name:
                     self.exclude_params.append(name)
                     break

        self.rng_dict = { name :   MultithreadedRNG(param.shape, seed=sampling_seed , threads = sampling_workers) for name , param in self.named_parameters()  }

    def regular_forward(self,x , target= None , return_probs = False):
            x = self.core_network(x)
            probs = F.log_softmax(x,dim=1)

            if return_probs:
                return probs
            else:
                loss = F.nll_loss(probs, target)

                return loss

    def ball_forward_alt(self,x,target):


        loss_list = []
        with torch.no_grad():
            for i in range(len(self.v)):

                for name,param in self.named_parameters():
                    if name in self.v[i].keys():

                        param.data = param.data + self.v[i][name].to(self.device)

                loss_list.append(self.regular_forward(x,target=target,return_probs=False))


                for name,param in self.named_parameters():
                    if name in self.v[i].keys():

                        param.data = param.data - self.v[i][name].to(self.device)


            idx = np.argmax(np.array(loss_list))

        for name, param in self.named_parameters():
            if name in self.v[idx].keys():
                param.data = param.data + self.v[idx][name].to(self.device)

        loss = self.regular_forward(x,target=target,return_probs=False)

        with torch.no_grad():
            for name,param in self.named_parameters():
                if name in self.v[idx].keys():
                    param.data = param.data - self.v[idx][name].to(self.device)

        return loss




    def forward(self, x , target=None, return_probs = False , train = True ):

        if train and len(self.v)>0:
            return self.ball_forward_alt(x,target)
        if train and len(self.v)==0:
            return self.regular_forward(x,target=target,return_probs=return_probs)

        if not train:
            return self.regular_forward(x,target=target,return_probs=return_probs)





    #############################################################################################
    '''Assess loss over batch "x" for
    "num_dir" random points within a "gamma"
    neighborhood of the current point'''
    #############################################################################################

    def assess(self,x ,target, seed=0, num_dir = 10 , gamma = 2 , return_v = False , return_all_losses=False):
        losses = []
        pert_v = {}

        rng_dict = self.rng_dict

        with torch.no_grad():
            np.random.seed(seed)
            for k_ in range(num_dir):
                for name,param in self.named_parameters():
                    if name not in self.exclude_params:
                        pert_v[name] = rng_dict[name].fill().to(self.device)
                        pert_v[name] *= gamma /torch.norm(pert_v[name])
                        param.data = param.data + pert_v[name]

                loss = self.forward(x , target = target , train = False , return_probs=False)
                losses.append([ loss , pert_v ])

                for name,param in self.named_parameters():
                    if name not in self.exclude_params:
                        param.data = param.data - pert_v[name]




            losses = np.array(losses)

            if return_all_losses:
                return losses[:,0]

            elif return_v:
                idx = np.argmax(np.array(losses)[:,0])
                return losses[idx,1]

            else:
                return losses
