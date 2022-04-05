import re
import pandas as pd
import matplotlib.pyplot as plt
from numpy import mean, std, arange
from math import sqrt

# figureType = 1 -> print loss
# figureType = otherwise -> print accuracy

# language = 1 -> print in portuguese
# language = otherwise -> print in english

def plot_image(figureType=0,language=1):

    # 2 class experiment
    filename1 = 'experiment-results/results-dirichlet/centralized/result-client1'
    filename2 = 'experiment-results/results-dirichlet/centralized/result-client2'
    filename3 = 'experiment-results/results-dirichlet/centralized/result-client3'
    filename4 = 'experiment-results/results-dirichlet/centralized/result-client4'
    filename5 = 'experiment-results/results-dirichlet/centralized/result-client5'
    filename6 = 'experiment-results/results-dirichlet/centralized/result-client6'
    filename7 = 'experiment-results/results-dirichlet/centralized/result-client7'
    filename8 = 'experiment-results/results-dirichlet/centralized/result-client8'
    filename9 = 'experiment-results/results-dirichlet/centralized/result-client9'
    filename10 = 'experiment-results/results-dirichlet/centralized/result-client10'



    results1 = open(filename1, 'r')
    results2 = open(filename2, 'r')
    results3 = open(filename3, 'r')
    results4 = open(filename4, 'r')
    results5 = open(filename5, 'r')
    results6 = open(filename6, 'r')
    results7 = open(filename7, 'r')
    results8 = open(filename8, 'r')
    results9 = open(filename9, 'r')
    results10 = open(filename10, 'r')

    Lines1 = results1.readlines()
    Lines2 = results2.readlines()
    Lines3 = results3.readlines()
    Lines4 = results4.readlines()
    Lines5 = results5.readlines()
    Lines6 = results6.readlines()
    Lines7 = results7.readlines()
    Lines8 = results8.readlines()
    Lines9 = results9.readlines()
    Lines10 = results10.readlines()


    accuracies = [[],[],[],[],[],[],[],[],[],[]]

    if figureType == 1:
        for line in Lines1:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[0].append(float(line.split(':')[1][1:].split(' ')[0]))        
        for line in Lines2:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[1].append(float(line.split(':')[1][1:].split(' ')[0]))
        for line in Lines3:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[2].append(float(line.split(':')[1][1:].split(' ')[0]))
        for line in Lines4:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[3].append(float(line.split(':')[1][1:].split(' ')[0]))
        for line in Lines5:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[4].append(float(line.split(':')[1][1:].split(' ')[0]))
        for line in Lines6:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[5].append(float(line.split(':')[1][1:].split(' ')[0]))
        for line in Lines7:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[6].append(float(line.split(':')[1][1:].split(' ')[0]))
        for line in Lines8:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[7].append(float(line.split(':')[1][1:].split(' ')[0]))
        for line in Lines9:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[8].append(float(line.split(':')[1][1:].split(' ')[0]))
        for line in Lines10:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[9].append(float(line.split(':')[1][1:].split(' ')[0]))
    else:
        for line in Lines1:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line) :
                accuracies[0].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines2:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[1].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines3:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[2].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines4:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[3].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines5:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[4].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines6:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[5].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines7:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[6].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines8:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[7].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines9:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[8].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines10:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies[9].append(float(line.split(':')[2].split(' ')[1]))
        

    x1Mean = mean(accuracies,axis=0);
    x1Interval = std(accuracies,axis=0)*1.96/sqrt(10);
    x = arange(len(x1Mean))


    # 2 class experiment cluster
    filename1 = 'experiment-results/results-dirichlet/alpha-0.5/result-client1'
    filename2 = 'experiment-results/results-dirichlet/alpha-0.5/result-client2'
    filename3 = 'experiment-results/results-dirichlet/alpha-0.5/result-client3'
    filename4 = 'experiment-results/results-dirichlet/alpha-0.5/result-client4'
    filename5 = 'experiment-results/results-dirichlet/alpha-0.5/result-client5'
    filename6 = 'experiment-results/results-dirichlet/alpha-0.5/result-client6'
    filename7 = 'experiment-results/results-dirichlet/alpha-0.5/result-client7'
    filename8 = 'experiment-results/results-dirichlet/alpha-0.5/result-client8'
    filename9 = 'experiment-results/results-dirichlet/alpha-0.5/result-client9'
    filename10 = 'experiment-results/results-dirichlet/alpha-0.5/result-client10'

    results1 = open(filename1, 'r')
    results2 = open(filename2, 'r')
    results3 = open(filename3, 'r')
    results4 = open(filename4, 'r')
    results5 = open(filename5, 'r')
    results6 = open(filename6, 'r')
    results7 = open(filename7, 'r')
    results8 = open(filename8, 'r')
    results9 = open(filename9, 'r')
    results10 = open(filename10, 'r')

    Lines1 = results1.readlines()
    Lines2 = results2.readlines()
    Lines3 = results3.readlines()
    Lines4 = results4.readlines()
    Lines5 = results5.readlines()
    Lines6 = results6.readlines()
    Lines7 = results7.readlines()
    Lines8 = results8.readlines()
    Lines9 = results9.readlines()
    Lines10 = results10.readlines()


    accuracies2 = [[],[],[],[],[],[],[],[],[],[]]

    if figureType == 1:
        for line in Lines1:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[0].append(float(line.split(':')[1][1:].split(' ')[0]))        
        for line in Lines2:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[1].append(float(line.split(':')[1][1:].split(' ')[0]))
        for line in Lines3:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[2]
        for line in Lines4:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[3].append(float(line.split(':')[1][1:].split(' ')[0]))
        for line in Lines5:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[4].append(float(line.split(':')[1][1:].split(' ')[0]))
        for line in Lines6:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[5].append(float(line.split(':')[1][1:].split(' ')[0]))
        for line in Lines7:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[6].append(float(line.split(':')[1][1:].split(' ')[0]))
        for line in Lines8:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[7].append(float(line.split(':')[1][1:].split(' ')[0]))
        for line in Lines9:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[8].append(float(line.split(':')[1][1:].split(' ')[0]))
        for line in Lines10:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[9].append(float(line.split(':')[1][1:].split(' ')[0]))

    else:
        for line in Lines1:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[0].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines2:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[1].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines3:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[2].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines4:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[3].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines5:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[4].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines6:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[5].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines7:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[6].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines8:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[7].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines9:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[8].append(float(line.split(':')[2].split(' ')[1]))
        for line in Lines10:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies2[9].append(float(line.split(':')[2].split(' ')[1]))
        

    x2Mean = mean(accuracies2,axis=0);
    x2Interval = std(accuracies2,axis=0)*1.96/sqrt(10);



    if language == 1 and figureType == 1:
        plt.plot(x, x1Mean, 'r-', label='Aprendizado Federado Tradicional')
        plt.fill_between(x, x1Mean - x1Interval, x1Mean + x1Interval, color='r', alpha=0.2)    
        plt.plot(x, x2Mean, 'b-',label='Proposta Atual')
        plt.fill_between(x, x2Mean - x2Interval, x2Mean + x2Interval, color='b', alpha=0.2)
        plt.ylabel('Perda no Teste', fontsize=16)
        plt.xlabel('Época', fontsize=16)
        plt.legend()
        plt.show()

    elif language == 1 and figureType != 1:
        plt.plot(x, x1Mean, 'r-', label='Aprendizado Federado Tradicional')
        plt.fill_between(x, x1Mean - x1Interval, x1Mean + x1Interval, color='r', alpha=0.2)
        plt.plot(x, x2Mean, 'b-',label='Proposta Atual')
        plt.fill_between(x, x2Mean - x2Interval, x2Mean + x2Interval, color='b', alpha=0.2)
        plt.ylabel('Acurácia', fontsize=16)
        plt.xlabel('Época', fontsize=16)
        plt.legend()
        plt.show()

        
    elif language != 1 and figureType == 1:
        plt.plot(x, x1Mean, 'r-', label='Traditional Federated Learning')
        plt.fill_between(x, x1Mean - x1Interval, x1Mean + x1Interval, color='r', alpha=0.2)
        plt.plot(x, x2Mean, 'b-',label='Our Proposal')
        plt.fill_between(x, x2Mean - x2Interval, x2Mean + x2Interval, color='b', alpha=0.2)
        plt.ylabel('Test Loss', fontsize=16)
        plt.xlabel('Epoch', fontsize=16)
        plt.legend()
        plt.show()


    elif language != 1 and figureType != 1:
        plt.plot(x, x1Mean, 'r-', label='Traditional Federated Learning')
        plt.fill_between(x, x1Mean - x1Interval, x1Mean + x1Interval, color='r', alpha=0.2)
        plt.plot(x, x2Mean, 'b-',label='Our Proposal')
        plt.fill_between(x, x2Mean - x2Interval, x2Mean + x2Interval, color='b', alpha=0.2)
        plt.ylabel('Accuracy', fontsize=16)
        plt.xlabel('Epoch', fontsize=16)
        plt.legend()
        plt.show()

plot_image(0,0)



