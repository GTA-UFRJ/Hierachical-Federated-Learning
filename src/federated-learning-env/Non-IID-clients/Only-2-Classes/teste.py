import tensorflow as tf
from pickle import load
import numpy as np
from sys import argv
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from sklearn import preprocessing

# first class
x_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class0Train','rb')))
y_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class0TrainLabel','rb')))
x_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class0Test','rb')))
y_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class0TestLabel','rb')))

x_train1 = x_train1[:2500]
y_train1 = y_train1[:2500]
x_test1 = x_test1[:500]
y_test1 = y_test1[:500]


# second class
x_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class1Train','rb')))
y_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class1TrainLabel','rb')))
x_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class1Test','rb')))
y_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class1TestLabel','rb')))

x_train2 = x_train2[:2500]
y_train2 = y_train2[:2500]
x_test2 = x_test2[:500]
y_test2 = y_test2[:500]

# first class
x_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class2Train','rb')))
y_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class2TrainLabel','rb')))
x_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class2Test','rb')))
y_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class2TestLabel','rb')))

x_train3 = x_train1[2500:]
y_train3 = y_train1[2500:]
x_test3 = x_test1[500:]
y_test3 = y_test1[500:]


# second class
x_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class3Train','rb')))
y_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class3TrainLabel','rb')))
x_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class3Test','rb')))
y_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class3TestLabel','rb')))

x_train4 = x_train2[2500:]
y_train4 = y_train2[2500:]
x_test4 = x_test2[500:]
y_test4 = y_test2[500:]



# create the dataset
x_train = np.concatenate((x_train1,x_train2,x_test1,x_test2,x_train3,x_train4,x_test3,x_test4))
y_train = np.concatenate((y_train1,y_train2,y_test1,y_test2,y_train3,y_train4,y_test3,y_test4))

#x_train = np.concatenate((x_train3,x_train4,x_test3,x_test4))
#y_train = np.concatenate((y_train3,y_train4,y_test3,y_test4))

x_train, y_train = shuffle(x_train, y_train, random_state=47527)



model = keras.models.load_model('model-1')

loss, accuracy = model.evaluate(x_test3,  y_test3, verbose=2)

