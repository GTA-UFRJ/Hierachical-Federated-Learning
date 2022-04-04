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
# usage: python client-7.py 

import flwr as fl
import tensorflow as tf
from pickle import load, dump
import numpy as np
from sys import argv

# client configuration
serverPort = '8080'

if len(argv) >= 2:
    serverPort = argv[1]


# first class
x_data = np.asarray(load(open('../../../../datasets/CIFAR-10/direchlet-partition/client_7_samples','rb')))
y_data = np.asarray(load(open('../../../../datasets/CIFAR-10/direchlet-partition/client_7_class','rb')))

x_train = x_data[:int(len(x_data)*0.7)]
y_train = y_data[:int(len(x_data)*0.7)]
x_test = x_data[int(len(x_data)*0.7):]
y_test = y_data[int(len(x_data)*0.7):]


model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

class CifarClient(fl.client.NumPyClient):
	
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=5,steps_per_epoch=31)
#        with open('weights/client-7-weights','wb') as writeFile:
#            dump(model.get_weights(),writeFile)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)
        return loss, len(x_test), {"accuracy": accuracy}

	

fl.client.start_numpy_client("[::]:"+serverPort, client=CifarClient())


