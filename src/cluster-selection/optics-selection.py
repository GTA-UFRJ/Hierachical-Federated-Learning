# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformática e Automação (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#
#
#
#
#
#
#
# usage: python optics-selection.py <read_file> <-minimum-number-of-clusters-samples>


from pickle import load, dump
from sys import argv
from sklearn.cluster import OPTICS
from os import listdir


# check if the parameters are correct
if not argv[1]:
    print("missing read file");

dataList = [];

# receives an ordenated list of client data vectors
for clientFile in range(1,11):
    dataList.append(load(open(argv[1]+'/'+'client-'+str(clientFile)+'-weights','rb'))[261]);

# verifying the OPTICS hyperparameter
if argv[2] and argv[2].isalpha:
    clientCluster = OPTICS(min_samples=int(argv[2])).fit(dataList);
else:
    clientCluster = OPTICS(min_samples=2).fit(dataList);
    
# print the result
print(clientCluster.labels_)

    


