# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformática e Automação (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)


import flwr as fl
import tensorflow as tf
from pickle import load


# split the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train[25000:30000]
y_train = y_train[25000:30000]
x_test = x_test[5000:6000]
y_test = y_test[5000:6000]


model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

class CifarClient(fl.client.NumPyClient):
	
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=5,steps_per_epoch=512)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)
        return loss, len(x_test), {"accuracy": accuracy}

	

fl.client.start_numpy_client("[::]:8082", client=CifarClient())


