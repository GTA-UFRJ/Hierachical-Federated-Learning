import flwr as fl
import tensorflow as tf
from pickle import load, dump
import numpy as np
from sys import argv

serverPort = '8080'
localEpochs = 5

if len(argv) >= 2:
    serverPort = argv[1]

if len(argv) >= 3:
    localEpochs = int(argv[2])

# first class
x_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class8Train','rb')),dtype=np.float32)
y_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class8TrainLabel','rb')),dtype=np.float32)
x_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class8Test','rb')),dtype=np.float32)
y_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class8TestLabel','rb')),dtype=np.float32)

x_train1 = x_train1[2500:]
y_train1 = y_train1[2500:]
x_test1 = x_test1[500:]
y_test1 = y_test1[500:]


# second class
x_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class9Train','rb')),dtype=np.float32)
y_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class9TrainLabel','rb')),dtype=np.float32)
x_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class9Test','rb')),dtype=np.float32)
y_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class9TestLabel','rb')),dtype=np.float32)

x_train2 = x_train2[2500:]
y_train2 = y_train2[2500:]
x_test2 = x_test2[500:]
y_test2 = y_test2[500:]

# create the training data
x_train = np.concatenate((x_train1,x_train2))

# create the training label
y_train = np.concatenate((y_train1,y_train2))

# create the test data
x_test = np.concatenate((x_test1,x_test2))

# create the test label
y_test = np.concatenate((y_test1,y_test2))


model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

class CifarClient(fl.client.NumPyClient):
	
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=localEpochs,steps_per_epoch=31)
        with open('weights/client-10-weights','wb') as writeFile:
            dump(model.get_weights(),writeFile)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)
        return loss, len(x_test), {"accuracy": accuracy}

	
fl.client.start_numpy_client("[::]:"+serverPort, client=CifarClient())



