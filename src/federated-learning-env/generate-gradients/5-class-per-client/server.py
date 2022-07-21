import tensorflow as tf
import flwr as fl

strategy = fl.server.strategy.FedAvg(min_available_clients=10,min_fit_clients=10,fraction_fit=1.0,fraction_eval=1.0)

fl.server.start_server(config={"num_rounds": 1},server_address='[::]:8071',strategy=strategy)


