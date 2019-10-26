#############################################################################################
'''DEFINE CLASSIFIERS'''
#############################################################################################



from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Conv2d_Shift , Linear_Shift

from fast_random import MultithreadedRNG

class DRM_Net(nn.Module):
    def __init__(self  ,
                 v_max_len = 1 ,
                sampling_workers = 8,
                sampling_seed = 0 ,
                num_dir = 20,
                core_network = None,
                device = None ):
        super(DRM_Net, self).__init__()

        assert core_network is not None , "Need to specify a core_network. Try Simple_Net(num_classes=num_classes ,input_shape=input_shape)."

        self.core_network = core_network
        #self.simple_net = resnet20(num_classes=num_classes)
        #self.simple_net = Simple_Net(num_classes=num_classes ,input_shape=input_shape)
        self.device = device
        self.num_dir = num_dir
        self.v_max_len = v_max_len
        self.v = deque(maxlen=v_max_len)

        self.train_loss = []
        self.test_loss =[]
        self.train_acc=[]
        self.test_acc =[]
        self.train_steps =0

        self.rng_dict = { name :  { 'weight' : MultithreadedRNG(module.weight.shape, seed=sampling_seed , threads = sampling_workers) ,
                          'bias' : MultithreadedRNG(module.bias.shape, seed=sampling_seed , threads = sampling_workers) if module.bias is not None else 0 }
            for name , module in self.named_modules() if isinstance(module , (Linear_Shift, Conv2d_Shift)) }

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

                for name,module in self.named_modules():
                    if name in self.v[i].keys():

                        module.weight_shift = self.v[i][name]['weight'].to(self.device)
                        module.bias_shift = self.v[i][name]['bias'].to(self.device)




                loss_list.append(self.regular_forward(x,target=target,return_probs=False))


            idx = np.argmax(np.array(loss_list))

        for name, module in self.named_modules():
            if name in self.v[idx].keys():
                module.weight_shift = self.v[idx][name]['weight'].to(self.device)
                module.bias_shift = self.v[idx][name]['bias'].to(self.device)

        loss = self.regular_forward(x,target=target,return_probs=False)


        for name,module in self.named_modules():
            if name in self.v[0].keys():
                module.weight_shift = torch.tensor(0,device=self.device)
                module.bias_shift = torch.tensor(0,device=self.device)


        return loss




    def forward(self, x , target=None, return_probs = False , train = True ):

        if train and len(self.v)>0:
            return self.ball_forward_alt(x,target)
        if train and len(self.v)==0:
            return self.regular_forward(x,target=target,return_probs=return_probs)

        if not train:
            return self.regular_forward(x,target=target,return_probs=return_probs)
