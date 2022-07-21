import tensorflow as tf
import flwr as fl
from sys import argv

if len(argv) >= 2:
   serverPort = argv[1]


strategy = fl.server.strategy.FedAvg(min_available_clients=10,min_fit_clients=10,fraction_fit=1.0,fraction_eval=1.0)

fl.server.start_server(config={"num_rounds": 1},server_address="[::]:"+serverPort,strategy=strategy)


