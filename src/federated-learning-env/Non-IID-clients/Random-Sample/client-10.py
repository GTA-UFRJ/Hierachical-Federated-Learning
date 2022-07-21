import flwr as fl
import tensorflow as tf
import numpy as np
from pickle import load


x_train = np.asarray(load(open('../../../../datasets/CIFAR-10/IID-distribution/training-data/data/dataset-trainx-client10','rb')), dtype=np.float32)
y_train = np.asarray(load(open('../../../../datasets/CIFAR-10/IID-distribution/training-data/label/dataset-trainy-client10','rb')), dtype=np.float32)
x_test = np.asarray(load(open('../../../../datasets/CIFAR-10/IID-distribution/test-data/data/dataset-testx-client10','rb')), dtype=np.float32)
y_test = np.asarray(load(open('../../../../datasets/CIFAR-10/IID-distribution/test-data/label/dataset-testy-client10','rb')), dtype=np.float32)



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

	

fl.client.start_numpy_client("[::]:8085", client=CifarClient())


