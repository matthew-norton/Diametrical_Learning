#############################################################################################
'''GET DATA'''
#############################################################################################
    #path = '/home/matthewnorton/Documents/pytorch_data/'
import torch
import torchvision

from torchvision import transforms

import numpy as np
#############################################################################################
'''GET CIFAR10'''
#############################################################################################

def get_CIFAR10(path,
                exclude_list,
                batch_size_train,
                batch_size_test,
                percent_flip):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])



    #path = '/home/matthewnorton/Documents/pytorch_data/'
    class SubLoader(torchvision.datasets.CIFAR10):
        def __init__(self, *args, exclude_list=[], percent_flip = 1 ,  **kwargs):
            super(SubLoader, self).__init__(*args, **kwargs)
            np.random.seed(0)
            if exclude_list == []:
                return

            if self.train:
                labels = np.array(self.targets)

                exclude = np.array(exclude_list).reshape(1, -1)
                mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

                self.data = self.data[mask]
                self.targets = labels[mask].tolist()

                self.num_flip = int(percent_flip*len(self.targets))
                label_set = set(self.targets)
                for i in range(self.num_flip):
                    self.targets[i] = np.random.choice([j for j in label_set if j != self.targets[i] ])
            else:
                labels = np.array(self.targets)
                exclude = np.array(exclude_list).reshape(1, -1)
                mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

                self.data = self.data[mask]
                self.targets = labels[mask].tolist()

    #exclude_list = [i for i in range(3,10)]
    trainset = SubLoader(root=path, train=True, download=True, transform=transform_train , exclude_list = exclude_list, percent_flip = percent_flip)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)

    testset = SubLoader(root=path, train=False, download=True, transform=transform_test , exclude_list = exclude_list)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=2)


    return train_loader, test_loader


#############################################################################################
'''GET MNIST'''
#############################################################################################




def get_MNIST(path,
                exclude_list,
                batch_size_train,
                batch_size_test,
                percent_flip):


    transform_train = transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])

    transform_test = transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])



    #path = '/home/matthewnorton/Documents/pytorch_data/'
    class SubLoader(torchvision.datasets.MNIST):
        def __init__(self, *args, exclude_list=[], percent_flip = 1 ,  **kwargs):
            super(SubLoader, self).__init__(*args, **kwargs)
            np.random.seed(0)
            if exclude_list == []:
                return

            if self.train:
                labels = np.array(self.targets)

                exclude = np.array(exclude_list).reshape(1, -1)
                mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

                self.data = self.data[mask]
                self.targets = labels[mask].tolist()

                self.num_flip = int(percent_flip*len(self.targets))
                label_set = set(self.targets)
                for i in range(self.num_flip):
                    self.targets[i] = np.random.choice([j for j in label_set if j != self.targets[i] ])
            else:
                labels = np.array(self.targets)
                exclude = np.array(exclude_list).reshape(1, -1)
                mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

                self.data = self.data[mask]
                self.targets = labels[mask].tolist()

    #exclude_list = [i for i in range(3,10)]
    trainset = SubLoader(root=path, train=True, download=True, transform=transform_train , exclude_list = exclude_list, percent_flip = percent_flip)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)

    testset = SubLoader(root=path, train=False, download=True, transform=transform_test , exclude_list = exclude_list)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=2)

    return train_loader, test_loader
