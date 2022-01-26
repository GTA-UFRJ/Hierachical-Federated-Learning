# Authors: Gustavo Franco Camilo and Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformática e Automação (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#
#
#
#
#
#
#
# usage: python split_iid.py <number-of-clients>

import tensorflow as tf
import pickle
import numpy as np
from sys import argv

def main(n_clients):
    
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train_list = x_train.tolist() 
    y_train_list = y_train.tolist() 
    x_test_list = x_test.tolist() 
    y_test_list = y_test.tolist() 
    train_length = len(x_train)
    test_length = len(x_test)

    trainx_filenames, trainy_filenames, testx_filenames, testy_filenames = generate_filenames(n_clients)

    for i in range(0,n_clients):
        samples_xtrain = []
        samples_ytrain = []
        for j in range(0,round(train_length/(n_clients))):
            random_selection = np.random.randint(0, high=(len(x_train_list)))
            sample_x = x_train_list.pop(random_selection)
            sample_y = y_train_list.pop(random_selection)
            samples_xtrain.append(sample_x)
            samples_ytrain.append(sample_y)
        pickle.dump(samples_xtrain,open(trainx_filenames[i],"wb"))
        pickle.dump(samples_ytrain,open(trainy_filenames[i],"wb"))
        samples_xtrain = []
        samples_ytrain =[]

        samples_xtest = []
        samples_ytest = []
        for k in range(0,round(test_length/(n_clients))):
            random_selection = np.random.randint(0, high=(len(x_test_list)))
            sample_x = x_test_list.pop(random_selection)
            sample_y = y_test_list.pop(random_selection)
            samples_xtest.append(sample_x)
            samples_ytest.append(sample_y)
        pickle.dump(samples_xtest,open(testx_filenames[i],"wb"))
        pickle.dump(samples_ytest,open(testy_filenames[i],"wb"))
        samples_xtest = []
        samples_ytest = []


def generate_filenames(n_clients):
    trainx_filenames = []
    trainy_filenames = []
    testx_filenames = []
    testy_filenames = []
    for i in range(0,n_clients):
        trainx_filenames.append('dataset-trainx-client' + str(i+1))
        trainy_filenames.append('dataset-trainy-client' + str(i++1))
        testx_filenames.append('dataset-testx-client' + str(i+1))
        testy_filenames.append('dataset-testy-client' + str(i+1))
    return trainx_filenames, trainy_filenames, testx_filenames, testy_filenames

if len(argv) != 2:
    print("Incorrect number of arguments. Usage: python3 split_iid.py [NUMBER_OF_CLIENTS]")
elif not argv[1].isnumeric():
    print("Incorrect argument. NUMBER_OF_CLIENTS must be an interger")
else:
    main(int(argv[1]))

