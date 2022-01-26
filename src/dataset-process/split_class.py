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
# usage: python split_class.py <number-of-clients>

import tensorflow as tf
import pickle
import numpy as np

def main():
    # classes in the dataset
    class0Train = []
    class1Train = []
    class2Train = []
    class3Train = []
    class4Train = []
    class5Train = []
    class6Train = []
    class7Train = []
    class8Train = []
    class9Train = []
    
    class0Test = []
    class1Test = []
    class2Test = []
    class3Test = []
    class4Test = []
    class5Test = []
    class6Test = []
    class7Test = []
    class8Test = []
    class9Test = []

    class0TrainLabel = []
    class1TrainLabel = []
    class2TrainLabel = []
    class3TrainLabel = []
    class4TrainLabel = []
    class5TrainLabel = []
    class6TrainLabel = []
    class7TrainLabel = []
    class8TrainLabel = []
    class9TrainLabel = []
    
    class0TestLabel = []
    class1TestLabel = []
    class2TestLabel = []
    class3TestLabel = []
    class4TestLabel = []
    class5TestLabel = []
    class6TestLabel = []
    class7TestLabel = []
    class8TestLabel = []
    class9TestLabel = []

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train_list = x_train.tolist() 
    y_train_list = y_train.tolist() 
    x_test_list = x_test.tolist() 
    y_test_list = y_test.tolist() 
    train_length = len(x_train)
    test_length = len(x_test)

    index = 0
    for sample in y_train_list:
        if sample[0] == 0:
            class0Train.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class0TrainLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 1:
            class1Train.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class1TrainLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 2:
            class2Train.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class2TrainLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 3:
            class3Train.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class3TrainLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 4:
            class4Train.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class4TrainLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 5:
            class5Train.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class5TrainLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 6:
            class6Train.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class6TrainLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 7:
            class7Train.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class7TrainLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 8:
            class8Train.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class8TrainLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 9:
            class9Train.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class9TrainLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1

    index = 0
    for sample in y_test_list:
        if sample[0] == 0:
            class0Test.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class0TestLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 1:
            class1Test.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class1TestLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 2:
            class2Test.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class2TestLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 3:
            class3Test.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class3TestLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 4:
            class4Test.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class4TestLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 5:
            class5Test.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class5TestLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 6:
            class6Test.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class6TestLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 7:
            class7Test.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class7TestLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 8:
            class8Test.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class8TestLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1
        elif sample[0] == 9:
            class9Test.append(np.asarray(x_train_list[index],dtype=np.uint8))
            class9TestLabel.append(np.asarray(sample,dtype=np.uint8))
            index += 1

    pickle.dump(class0Train,open('class0Train',"wb"))
    pickle.dump(class1Train,open('class1Train',"wb"))
    pickle.dump(class2Train,open('class2Train',"wb"))
    pickle.dump(class3Train,open('class3Train',"wb"))
    pickle.dump(class4Train,open('class4Train',"wb"))
    pickle.dump(class5Train,open('class5Train',"wb"))
    pickle.dump(class6Train,open('class6Train',"wb"))
    pickle.dump(class7Train,open('class7Train',"wb"))
    pickle.dump(class8Train,open('class8Train',"wb"))
    pickle.dump(class9Train,open('class9Train',"wb"))

    pickle.dump(class0Test,open('class0Test',"wb"))
    pickle.dump(class1Test,open('class1Test',"wb"))
    pickle.dump(class2Test,open('class2Test',"wb"))
    pickle.dump(class3Test,open('class3Test',"wb"))
    pickle.dump(class4Test,open('class4Test',"wb"))
    pickle.dump(class5Test,open('class5Test',"wb"))
    pickle.dump(class6Test,open('class6Test',"wb"))
    pickle.dump(class7Test,open('class7Test',"wb"))
    pickle.dump(class8Test,open('class8Test',"wb"))
    pickle.dump(class9Test,open('class9Test',"wb"))

    pickle.dump(class0TrainLabel,open('class0TrainLabel',"wb"))
    pickle.dump(class1TrainLabel,open('class1TrainLabel',"wb"))
    pickle.dump(class2TrainLabel,open('class2TrainLabel',"wb"))
    pickle.dump(class3TrainLabel,open('class3TrainLabel',"wb"))
    pickle.dump(class4TrainLabel,open('class4TrainLabel',"wb"))
    pickle.dump(class5TrainLabel,open('class5TrainLabel',"wb"))
    pickle.dump(class6TrainLabel,open('class6TrainLabel',"wb"))
    pickle.dump(class7TrainLabel,open('class7TrainLabel',"wb"))
    pickle.dump(class8TrainLabel,open('class8TrainLabel',"wb"))
    pickle.dump(class9TrainLabel,open('class9TrainLabel',"wb"))

    pickle.dump(class0TestLabel,open('class0TestLabel',"wb"))
    pickle.dump(class1TestLabel,open('class1TestLabel',"wb"))
    pickle.dump(class2TestLabel,open('class2TestLabel',"wb"))
    pickle.dump(class3TestLabel,open('class3TestLabel',"wb"))
    pickle.dump(class4TestLabel,open('class4TestLabel',"wb"))
    pickle.dump(class5TestLabel,open('class5TestLabel',"wb"))
    pickle.dump(class6TestLabel,open('class6TestLabel',"wb"))
    pickle.dump(class7TestLabel,open('class7TestLabel',"wb"))
    pickle.dump(class8TestLabel,open('class8TestLabel',"wb"))
    pickle.dump(class9TestLabel,open('class9TestLabel',"wb"))


main()

