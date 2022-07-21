import flwr as fl
import tensorflow as tf
from pickle import load
import numpy as np
from sys import argv
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

cluster = 1
serverPort = '8080'

if len(argv) >= 2:
    serverPort = argv[1]

classes = 10

if len(argv) >= 3:
    classes = int(argv[2])

if cluster == 1:
    # first class
    x_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class0Train','rb')))
    y_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class0TrainLabel','rb')))
    x_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class0Test','rb')))
    y_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class0TestLabel','rb')))
    
    x_train1 = x_train1[:1000]
    y_train1 = y_train1[:1000]
    x_test1 = x_test1[:100]
    y_test1 = y_test1[:100]
    
    
    # second class
    x_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class1Train','rb')))
    y_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class1TrainLabel','rb')))
    x_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class1Test','rb')))
    y_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class1TestLabel','rb')))
    
    x_train2 = x_train2[:1000]
    y_train2 = y_train2[:1000]
    x_test2 = x_test2[:100]
    y_test2 = y_test2[:100]
    
    # third class
    
    x_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class2Train','rb')))
    y_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class2TrainLabel','rb')))
    x_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class2Test','rb')))
    y_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class2TestLabel','rb')))
    
    x_train3 = x_train3[:1000]
    y_train3 = y_train3[:1000]
    x_test3 = x_test3[:100]
    y_test3 = y_test3[:100]
    
    
    # fourth class
    
    x_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class3Train','rb')))
    y_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class3TrainLabel','rb')))
    x_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class3Test','rb')))
    y_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class3TestLabel','rb')))
    
    x_train4 = x_train4[:1000]
    y_train4 = y_train4[:1000]
    x_test4 = x_test4[:100]
    y_test4 = y_test4[:100]
    
    # fifth class
    
    x_train5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class4Train','rb')))
    y_train5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class4TrainLabel','rb')))
    x_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class4Test','rb')))
    y_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class4TestLabel','rb')))
    
    x_train5 = x_train5[:1000]
    y_train5 = y_train5[:1000]
    x_test5 = x_test5[:100]
    y_test5 = y_test5[:100]

else:
    # first class
    x_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class5Train','rb')),dtype=np.float32)
    y_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class5TrainLabel','rb')),dtype=np.float32)
    x_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class5Test','rb')),dtype=np.float32)
    y_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class5TestLabel','rb')),dtype=np.float32)
    
    x_train1 = x_train1[:1000]
    y_train1 = y_train1[:1000]
    x_test1 = x_test1[:100]
    y_test1 = y_test1[:100]
    
    
    # second class
    x_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class6Train','rb')),dtype=np.float32)
    y_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class6TrainLabel','rb')),dtype=np.float32)
    x_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class6Test','rb')),dtype=np.float32)
    y_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class6TestLabel','rb')),dtype=np.float32)
    
    x_train2 = x_train2[:1000]
    y_train2 = y_train2[:1000]
    x_test2 = x_test2[:100]
    y_test2 = y_test2[:100]
    
    # third class
    
    x_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class7Train','rb')),dtype=np.float32)
    y_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class7TrainLabel','rb')),dtype=np.float32)
    x_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class7Test','rb')),dtype=np.float32)
    y_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class7TestLabel','rb')),dtype=np.float32)
    
    x_train3 = x_train3[:1000]
    y_train3 = y_train3[:1000]
    x_test3 = x_test3[:100]
    y_test3 = y_test3[:100]
    
    
    # fourth class
    
    x_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class8Train','rb')),dtype=np.float32)
    y_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class8TrainLabel','rb')),dtype=np.float32)
    x_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class8Test','rb')),dtype=np.float32)
    y_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class8TestLabel','rb')),dtype=np.float32)
    
    x_train4 = x_train4[:1000]
    y_train4 = y_train4[:1000]
    x_test4 = x_test4[:100]
    y_test4 = y_test4[:100]
    
    # fifth class
    
    x_train5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class9Train','rb')),dtype=np.float32)
    y_train5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class9TrainLabel','rb')),dtype=np.float32)
    x_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class9Test','rb')),dtype=np.float32)
    y_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class9TestLabel','rb')),dtype=np.float32)
    
    x_train5 = x_train5[:1000]
    y_train5 = y_train5[:1000]
    x_test5 = x_test5[:100]
    y_test5 = y_test5[:100]






x_train = np.concatenate((x_train1,x_train2,x_train3,x_train4,x_train5,x_test1,x_test2,x_test3,x_test4,x_test5))
y_train = np.concatenate((y_train1,y_train2,y_train3,y_train4,y_train5,y_test1,y_test2,y_test3,y_test4,y_test5))

x_train, y_train = shuffle(x_train, y_train, random_state=47527)

#if cluster != 1:
#    y_train = y_train-5

model1 = tf.keras.models.load_model('weights-1')
model = tf.keras.models.load_model('weights-6')

print(len(model1.layers))

for i in range(11):
    if len(model.layers[i].weights) != 0:
        model.layers[i].weights[0] = (model1.layers[i].weights[0] + model.layers[i].weights[0])
        model.layers[i].weights[1] = (model1.layers[i].weights[1] + model.layers[i].weights[1])

loss, accuracy = model.evaluate(x_train[2100:],  y_train[2100:], verbose=2)
