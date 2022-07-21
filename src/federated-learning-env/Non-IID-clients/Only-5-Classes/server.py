import tensorflow as tf
import flwr as fl
from sys import argv

serverPort = '8080'

if len(argv) >= 2:
    serverPort = argv[1]

strategy = fl.server.strategy.FedAvg(min_available_clients=5,fraction_fit=0.5,fraction_eval=1.0)

fl.server.start_server(config={"num_rounds": 25},server_address='[::]:'+serverPort,strategy=strategy)


