import tensorflow as tf
import flwr as fl

strategy = fl.server.strategy.FedAvg(min_available_clients=5,fraction_fit=0.2,fraction_eval=1.0)

fl.server.start_server(config={"num_rounds": 400},strategy=strategy)


