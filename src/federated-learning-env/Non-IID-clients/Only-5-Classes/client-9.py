import flwr as fl
import tensorflow as tf
from pickle import load
import numpy as np


# first class
x_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class5Train','rb')),dtype=np.float32)
y_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class5TrainLabel','rb')),dtype=np.float32)
x_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class5Test','rb')),dtype=np.float32)
y_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class5TestLabel','rb')),dtype=np.float32)

x_train1 = x_train1[3000:4000]
y_train1 = y_train1[3000:4000]
x_test1 = x_test1[300:400]
y_test1 = y_test1[300:400]


# second class
x_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class6Train','rb')),dtype=np.float32)
y_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class6TrainLabel','rb')),dtype=np.float32)
x_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class6Test','rb')),dtype=np.float32)
y_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class6TestLabel','rb')),dtype=np.float32)

x_train2 = x_train2[3000:4000]
y_train2 = y_train2[3000:4000]
x_test2 = x_test2[300:400]
y_test2 = y_test2[300:400]

# third class

x_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class7Train','rb')),dtype=np.float32)
y_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class7TrainLabel','rb')),dtype=np.float32)
x_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class7Test','rb')),dtype=np.float32)
y_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class7TestLabel','rb')),dtype=np.float32)

x_train3 = x_train3[3000:4000]
y_train3 = y_train3[3000:4000]
x_test3 = x_test3[300:400]
y_test3 = y_test3[300:400]


# fourth class

x_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class8Train','rb')),dtype=np.float32)
y_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class8TrainLabel','rb')),dtype=np.float32)
x_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class8Test','rb')),dtype=np.float32)
y_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class8TestLabel','rb')),dtype=np.float32)

x_train4 = x_train4[3000:4000]
y_train4 = y_train4[3000:4000]
x_test4 = x_test4[300:400]
y_test4 = y_test4[300:400]

# fifth class

x_train5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class9Train','rb')),dtype=np.float32)
y_train5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class9TrainLabel','rb')),dtype=np.float32)
x_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class9Test','rb')),dtype=np.float32)
y_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class9TestLabel','rb')),dtype=np.float32)

x_train5 = x_train5[3000:4000]
y_train5 = y_train5[3000:4000]
x_test5 = x_test5[300:400]
y_test5 = y_test5[300:400]


# create the training data
np.append(x_train1,x_train2)
np.append(x_train1,x_train3)
np.append(x_train1,x_train4)
np.append(x_train1,x_train5)

# create the training label
np.append(y_train1,y_train2)
np.append(y_train1,y_train3)
np.append(y_train1,y_train4)
np.append(y_train1,y_train5)

# create the test data
np.append(x_test1,x_test2)
np.append(x_test1,x_test3)
np.append(x_test1,x_test4)
np.append(x_test1,x_test5)

# create the test label
np.append(y_test1,y_test2)
np.append(y_test1,y_test3)
np.append(y_test1,y_test4)
np.append(y_test1,y_test5)

x_train = x_train1
y_train = y_train1
x_test = x_test1
y_test = y_test1


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


