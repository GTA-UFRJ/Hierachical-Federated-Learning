import flwr as fl
import tensorflow as tf
from pickle import load, dump
import numpy as np
from sys import argv


# client configuration
clientID = 6
serverPort = '8080'
numberOfClients = 10
trainLength = 5000
testLength = 1000
alfa = 0.55



if len(argv) >= 2:
    alfa = float(argv[1])

if len(argv) >= 3:
    serverPort = argv[2]

if len(argv) == 4:
    numberOfClients = argv[3] 


majorityTrain = int(trainLength*alfa)
majorityTest = int(testLength*alfa)
minorityTrain = int(trainLength*((1-alfa)/9))
minorityTest = int(testLength*((1-alfa)/9))


# load the dataset

# first class
x_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class0Train','rb')),dtype=np.float32)
y_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class0TrainLabel','rb')),dtype=np.float32)
x_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class0Test','rb')),dtype=np.float32)
y_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class0TestLabel','rb')),dtype=np.float32)

x_train1 = x_train1[majorityTrain+minorityTrain*(int(clientID)-2):majorityTrain+minorityTrain*(int(clientID)-1)+1]
y_train1 = y_train1[majorityTrain+minorityTrain*(int(clientID)-2):majorityTrain+minorityTrain*(int(clientID)-1)+1]
x_test1 = x_test1[majorityTest+minorityTest*(int(clientID)-2):majorityTest+minorityTest*(int(clientID)-1)+1]
y_test1 = y_test1[majorityTest+minorityTest*(int(clientID)-2):majorityTest+minorityTest*(int(clientID)-1)+1]


# second class
x_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class1Train','rb')),dtype=np.float32)
y_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class1TrainLabel','rb')),dtype=np.float32)
x_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class1Test','rb')),dtype=np.float32)
y_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class1TestLabel','rb')),dtype=np.float32)

x_train2 = x_train2[majorityTrain+minorityTrain*(int(clientID)-2):majorityTrain+minorityTrain*(int(clientID)-1)+1]
y_train2 = y_train2[majorityTrain+minorityTrain*(int(clientID)-2):majorityTrain+minorityTrain*(int(clientID)-1)+1]
x_test2 = x_test2[majorityTest+minorityTest*(int(clientID)-2):majorityTest+minorityTest*(int(clientID)-1)+1]
y_test2 = y_test2[majorityTest+minorityTest*(int(clientID)-2):majorityTest+minorityTest*(int(clientID)-1)+1]

# third class

x_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class2Train','rb')),dtype=np.float32)
y_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class2TrainLabel','rb')),dtype=np.float32)
x_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class2Test','rb')),dtype=np.float32)
y_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class2TestLabel','rb')),dtype=np.float32)

x_train3 = x_train3[majorityTrain+minorityTrain*(int(clientID)-2):majorityTrain+minorityTrain*(int(clientID)-1)+1]
y_train3 = y_train3[majorityTrain+minorityTrain*(int(clientID)-2):majorityTrain+minorityTrain*(int(clientID)-1)+1]
x_test3 = x_test3[majorityTest+minorityTest*(int(clientID)-2):majorityTest+minorityTest*(int(clientID)-1)+1]
y_test3 = y_test3[majorityTest+minorityTest*(int(clientID)-2):majorityTest+minorityTest*(int(clientID)-1)+1]


# fourth class

x_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class3Train','rb')),dtype=np.float32)
y_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class3TrainLabel','rb')),dtype=np.float32)
x_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class3Test','rb')),dtype=np.float32)
y_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class3TestLabel','rb')),dtype=np.float32)

x_train4 = x_train4[majorityTrain+minorityTrain*(int(clientID)-2):majorityTrain+minorityTrain*(int(clientID)-1)+1]
y_train4 = y_train4[majorityTrain+minorityTrain*(int(clientID)-2):majorityTrain+minorityTrain*(int(clientID)-1)+1]
x_test4 = x_test4[majorityTest+minorityTest*(int(clientID)-2):majorityTest+minorityTest*(int(clientID)-1)+1]
y_test4 = y_test4[majorityTest+minorityTest*(int(clientID)-2):majorityTest+minorityTest*(int(clientID)-1)+1]

# fifth class

x_train5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class4Train','rb')),dtype=np.float32)
y_train5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class4TrainLabel','rb')),dtype=np.float32)
x_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class4Test','rb')),dtype=np.float32)
y_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class4TestLabel','rb')),dtype=np.float32)

x_train5 = x_train5[majorityTrain+minorityTrain*(int(clientID)-2):majorityTrain+minorityTrain*(int(clientID)-1)+1]
y_train5 = y_train5[majorityTrain+minorityTrain*(int(clientID)-2):majorityTrain+minorityTrain*(int(clientID)-1)+1]
x_test5 = x_test5[majorityTest+minorityTest*(int(clientID)-2):majorityTest+minorityTest*(int(clientID)-1)+1]
y_test5 = y_test5[majorityTest+minorityTest*(int(clientID)-2):majorityTest+minorityTest*(int(clientID)-1)+1]

# sixth class
x_train6 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class5Train','rb')),dtype=np.float32)
y_train6 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class5TrainLabel','rb')),dtype=np.float32)
x_test6 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class5Test','rb')),dtype=np.float32)
y_test6 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class5TestLabel','rb')),dtype=np.float32)

x_train6 = x_train6[minorityTrain*(int(clientID)-1):majorityTrain+minorityTrain*(int(clientID)-1)]
y_train6 = y_train6[minorityTrain*(int(clientID)-1):majorityTrain+minorityTrain*(int(clientID)-1)]
x_test6 = x_test6[minorityTest*(int(clientID)-1):majorityTest+minorityTest*(int(clientID)-1)]
y_test6 = y_test6[minorityTest*(int(clientID)-1):majorityTest+minorityTest*(int(clientID)-1)]


# seventh class
x_train7 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class6Train','rb')),dtype=np.float32)
y_train7 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class6TrainLabel','rb')),dtype=np.float32)
x_test7 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class6Test','rb')),dtype=np.float32)
y_test7 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class6TestLabel','rb')),dtype=np.float32)

x_train7 = x_train7[minorityTrain*(int(clientID)-1):minorityTrain*int(clientID)+1]
y_train7 = y_train7[minorityTrain*(int(clientID)-1):minorityTrain*int(clientID)+1]
x_test7 = x_test7[minorityTest*(int(clientID)-1):minorityTest*int(clientID)+1]
y_test7 = y_test7[minorityTest*(int(clientID)-1):minorityTest*int(clientID)+1]

# third class

x_train8 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class7Train','rb')),dtype=np.float32)
y_train8 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class7TrainLabel','rb')),dtype=np.float32)
x_test8 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class7Test','rb')),dtype=np.float32)
y_test8 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class7TestLabel','rb')),dtype=np.float32)

x_train8 = x_train8[minorityTrain*(int(clientID)-1):minorityTrain*int(clientID)+1]
y_train8 = y_train8[minorityTrain*(int(clientID)-1):minorityTrain*int(clientID)+1]
x_test8 = x_test8[minorityTest*(int(clientID)-1):minorityTest*int(clientID)+1]
y_test8 = y_test8[minorityTest*(int(clientID)-1):minorityTest*int(clientID)+1]


# fourth class

x_train9 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class8Train','rb')),dtype=np.float32)
y_train9 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class8TrainLabel','rb')),dtype=np.float32)
x_test9 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class8Test','rb')),dtype=np.float32)
y_test9 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class8TestLabel','rb')),dtype=np.float32)

x_train9 = x_train9[minorityTrain*(int(clientID)-1):minorityTrain*int(clientID)+1]
y_train9 = y_train9[minorityTrain*(int(clientID)-1):minorityTrain*int(clientID)+1]
x_test9 = x_test9[minorityTest*(int(clientID)-1):minorityTest*int(clientID)+1]
y_test9 = y_test9[minorityTest*(int(clientID)-1):minorityTest*int(clientID)+1]

# fifth class

x_train10 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class9Train','rb')),dtype=np.float32)
y_train10 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class9TrainLabel','rb')),dtype=np.float32)
x_test10 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class9Test','rb')),dtype=np.float32)
y_test10 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class9TestLabel','rb')),dtype=np.float32)

x_train10 = x_train10[minorityTrain*(int(clientID)-1):minorityTrain*int(clientID)+1]
y_train10 = y_train10[minorityTrain*(int(clientID)-1):minorityTrain*int(clientID)+1]
x_test10 = x_test10[minorityTest*(int(clientID)-1):minorityTest*int(clientID)+1]
y_test10 = y_test10[minorityTest*(int(clientID)-1):minorityTest*int(clientID)+1]

# create the training data
x_train = np.concatenate((x_train1,x_train2,x_train3,x_train4,x_train5,x_train6,x_train7,x_train8,x_train9,x_train10))

# create the training label
y_train = np.concatenate((y_train1,y_train2,y_train3,y_train4,y_train5,y_train6,y_train7,y_train8,y_train9,y_train10))

# create the test data
x_test = np.concatenate((x_test1,x_test2,x_test3,x_test4,x_test5,x_test6,x_test7,x_test8,x_test9,x_test10))

# create the test label
y_test = np.concatenate((y_test1,y_test2,y_test3,y_test4,y_test5,y_test6,y_test7,y_test8,y_test9,y_test10))


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
        with open('weights/client-'+str(clientID)+'-weights','wb') as writeFile:
            dump(model.get_weights(),writeFile)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)
        return loss, len(x_test), {"accuracy": accuracy}

	

fl.client.start_numpy_client("[::]:"+serverPort, client=CifarClient())


