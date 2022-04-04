import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import torchvision.transforms as transforms
import torch.utils.data as data


from datasets import MNIST_truncated, CIFAR10_truncated
from pickle import dump


number_of_clients = 10

def load_mnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)

def partition_data(dataset, datadir, logdir, partition, n_nets, alpha=0.5):

    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)

    X_train = np.concatenate((X_train,X_test))
    y_train = np.concatenate((y_train,y_test))
    
    n_train = X_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero-dir":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
				## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    print (traindata_cls_counts)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)

def record_net_data_stats(y_train, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logging.debug('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

X_train, y_train, X_test, y_test, data_index, counts = partition_data('cifar10','cifar10','logs','hetero-dir',number_of_clients)


client_data = [[],[]]

for index in range(number_of_clients):
    for item in data_index[index]:
        client_data[0].append(X_train[item])
        client_data[1].append(y_train[item])
        
    # save the current client dataset and clean the variable to the next client
    dump(client_data[0],open('../../datasets/CIFAR-10/direchlet-partition/client_'+str(index+1)+'_samples',"wb"))
    dump(client_data[1],open('../../datasets/CIFAR-10/direchlet-partition/client_'+str(index+1)+'_class',"wb"))

    client_data = [[],[]]
        

