import flwr as fl
import tensorflow as tf
from pickle import load
import numpy as np


# first class
x_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class0Train','rb')),dtype=np.float32)
y_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class0TrainLabel','rb')),dtype=np.float32)
x_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class0Test','rb')),dtype=np.float32)
y_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class0TestLabel','rb')),dtype=np.float32)

x_train1 = x_train1[:1000]
y_train1 = y_train1[:1000]
x_test1 = x_test1[:100]
y_test1 = y_test1[:100]


# second class
x_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class1Train','rb')),dtype=np.float32)
y_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class1TrainLabel','rb')),dtype=np.float32)
x_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class1Test','rb')),dtype=np.float32)
y_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class1TestLabel','rb')),dtype=np.float32)

x_train2 = x_train2[:1000]
y_train2 = y_train2[:1000]
x_test2 = x_test2[:100]
y_test2 = y_test2[:100]

# third class

x_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class2Train','rb')),dtype=np.float32)
y_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class2TrainLabel','rb')),dtype=np.float32)
x_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class2Test','rb')),dtype=np.float32)
y_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class2TestLabel','rb')),dtype=np.float32)

x_train3 = x_train3[:1000]
y_train3 = y_train3[:1000]
x_test3 = x_test3[:100]
y_test3 = y_test3[:100]


# fourth class

x_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class3Train','rb')),dtype=np.float32)
y_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class3TrainLabel','rb')),dtype=np.float32)
x_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class3Test','rb')),dtype=np.float32)
y_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class3TestLabel','rb')),dtype=np.float32)

x_train4 = x_train4[:1000]
y_train4 = y_train4[:1000]
x_test4 = x_test4[:100]
y_test4 = y_test4[:100]

# fifth class

x_train5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class4Train','rb')),dtype=np.float32)
y_train5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class4TrainLabel','rb')),dtype=np.float32)
x_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class4Test','rb')),dtype=np.float32)
y_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class4TestLabel','rb')),dtype=np.float32)

x_train5 = x_train5[:1000]
y_train5 = y_train5[:1000]
x_test5 = x_test5[:100]
y_test5 = y_test5[:100]


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

	

fl.client.start_numpy_client("[::]:8081", client=CifarClient())


