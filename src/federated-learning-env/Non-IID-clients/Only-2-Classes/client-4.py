import flwr as fl
import tensorflow as tf
from pickle import load
import numpy as np
from sys import argv
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from sklearn import preprocessing

serverPort = '8080'

if len(argv) >= 2:
    serverPort = argv[1]


# first class
x_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class2Train','rb')))
y_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class2TrainLabel','rb')))
x_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class2Test','rb')))
y_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class2TestLabel','rb')))

x_train1 = x_train1[2500:]
y_train1 = y_train1[2500:]
x_test1 = x_test1[500:]
y_test1 = y_test1[500:]


# second class
x_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class3Train','rb')))
y_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class3TrainLabel','rb')))
x_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class3Test','rb')))
y_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class3TestLabel','rb')))

x_train2 = x_train2[2500:]
y_train2 = y_train2[2500:]
x_test2 = x_test2[500:]
y_test2 = y_test2[500:]



# create the dataset
x_train = np.concatenate((x_train1,x_train2,x_test1,x_test2))

y_train = np.concatenate((y_train1,y_train2,y_test1,y_test2))

x_train, y_train = shuffle(x_train, y_train, random_state=47527)


lb = preprocessing.LabelBinarizer()
lb.fit(y_train)
y_train = lb.transform(y_train)

model = keras.Sequential()


# Model adapted from https://towardsdatascience.com/10-minutes-to-building-a-cnn-binary-image-classifier-in-tensorflow-4e216b2034aa

# Convolutional layer and maxpool layer 1
model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3),padding='same'))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 2
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 3
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 4
model.add(keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(keras.layers.MaxPool2D(2,2))

# This layer flattens the resulting image array to 1D array
model.add(keras.layers.Flatten())

# Hidden layer with 512 neurons and Rectified Linear Unit activation function 
model.add(keras.layers.Dense(512,activation='relu'))

#Here we use sigmoid activation function which makes our model output to lie between 0 and 1
model.add(keras.layers.Dense(10,activation='sigmoid'))

# compile the model
model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(learning_rate=0.001),metrics='accuracy')


# federated client
class CifarClient(fl.client.NumPyClient):
	
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train[:2100],y_train[:2100],validation_split=0.2,batch_size=70,epochs=5,verbose=1)
        model.save('model-4')
        return model.get_weights(), len(x_train[:2100]), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_train[2100:],  y_train[2100:], verbose=2)
        return loss, len(x_train[2100:]), {"accuracy": accuracy}

	

fl.client.start_numpy_client("[::]:"+serverPort, client=CifarClient())
