from flautim.pytorch.common import run_federated, weighted_average
from flautim.pytorch import Model, Dataset
from flautim.pytorch.federated import Experiment
import sypak_dataset, r50_model, sypak_experiment
import flautim as fl
import flautim.metrics as flm
import flwr
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerConfig, ServerAppComponents
import pandas as pd
import numpy as np
from data_prep import dirichlet_partition
from torchvision.transforms import v2 as tranformsv2

train_transform = tranformsv2.Compose([
    tranformsv2.RandomRotation(15),
    tranformsv2.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

eval_transforms = tranformsv2.Compose([
    tranformsv2.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

num_clientes = 2
num_rounds = 20

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
    }
    return config

def generate_server_fn(context, eval_fn, **kwargs):

    def create_server_fn(context_flwr:  Context):

        net = r50_model.ResNet50Classifier(context, num_classes=3)
        params = ndarrays_to_parameters(net.get_parameters())

        strategy = flwr.server.strategy.FedAvg(
                        min_available_clients=2,
                        initial_parameters=params,
                        evaluate_metrics_aggregation_fn=weighted_average,
                        fraction_fit=1,  # 100% clients sampled each round to do fit()
                        fraction_evaluate=0.5,  # 50% clients sample each round to do evaluate()
                        evaluate_fn=eval_fn,
                        on_fit_config_fn = fit_config,
                        on_evaluate_config_fn = fit_config,
                        )
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(config=config, strategy=strategy)
    return create_server_fn

def generate_client_fn(context, files):

    def create_client_fn(context_flwr:  Context):
        cid = int(context_flwr.node_config["partition-id"])
        file = int(cid)
        #fl.log(f"Creating client {cid} with partition {file}...")
        model = r50_model.ResNet50Classifier(context, num_classes=3)
        dataset = sypak_dataset.SypakDataset(files[file], train_transforms=train_transform, eval_transforms=eval_transforms)

        return sypak_experiment.SypakExperiment(model, dataset, context).to_client()

    return create_client_fn

def evaluate_fn(context, files):
    def fn(server_round, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        The, the model will be evaluate on the test set (recall this is the
        whole MNIST test set)."""

        model = r50_model.ResNet50Classifier(context, num_classes=3, suffix="FL-Global")
        model.set_parameters(parameters)

        dataset = sypak_dataset.SypakDataset(files[0], train_transforms=train_transform, eval_transforms=eval_transforms)

        experiment = sypak_experiment.SypakExperiment(model, dataset, context)

        config["server_round"] = server_round

        loss, _, return_dic = experiment.evaluate(parameters, config)

        return loss, return_dic

    return fn

if __name__ == '__main__':

    context = fl.init()
    fl.log(f"Flautim Federated Experiment Started!!!")

    parquet_file = "./data/dataset_sypak_cropped.parquet"

    df = pd.read_parquet(parquet_file)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    #partitions = np.array_split(df, num_clientes+1)
    partitions = dirichlet_partition(df, num_clients=num_clientes+1, alpha=0.5)
    #fl.log(f"Data partitioned among {len(partitions)} clients.")
    client_fn_callback = generate_client_fn(context, partitions)
    evaluate_fn_callback = evaluate_fn(context, partitions)
    server_fn_callback = generate_server_fn(context, eval_fn = evaluate_fn_callback)
    run_federated(client_fn_callback, server_fn_callback, num_clients=num_clientes)