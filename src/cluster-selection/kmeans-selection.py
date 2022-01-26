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
# usage: python kmeans-selection.py <read_file_data> <number-of-clusters>


from pickle import load, dump
from sys import argv
from sklearn.cluster import KMeans

# check if the parameters are correct
if not argv[1]:
    print("missing read file");

dataList = [];

# receives an ordenated list of client data vectors
for clientFile in range(1,11):
    dataList.append(load(open(argv[1]+'/'+'client-'+str(clientFile)+'-weights','rb'))[261]);

# verifying the kmeans hyperparameter
if argv[2] and argv[2].isnumeric:
    clientCluster = KMeans(n_clusters=int(argv[2]), random_state=0).fit(dataList)
else:
    clientCluster = KMeans(n_clusters=5, random_state=0).fit(dataList)
    
# print the clusterization result
print(clientCluster.labels_)






