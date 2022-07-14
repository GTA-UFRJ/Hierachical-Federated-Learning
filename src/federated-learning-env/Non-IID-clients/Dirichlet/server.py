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
# usage: python server.py <numbers-of-rounds>  <server-port>

import tensorflow as tf
import flwr as fl
from sys import argv

numRounds = 400
serverPort ='8080'

if len(argv) >= 2:
    numRounds = int(argv[1])

if len(argv) >= 3:
    serverPort = argv[2]

strategy = fl.server.strategy.FedAvg(min_available_clients=2,fraction_fit=0.5,fraction_eval=1.0)

fl.server.start_server(config={"num_rounds": numRounds},server_address='[::]:'+serverPort,strategy=strategy)



