import flautim as fl
import sypak_dataset, r50_model, sypak_experiment
import numpy as np
import pandas as pd
import flautim.metrics as flm
from pathlib import Path
from data_prep import dirichlet_partition
import sklearn.metrics as skmetrics
from torchvision.transforms import v2 as tranformsv2

train_transform = tranformsv2.Compose([
    tranformsv2.RandomRotation(15),
    tranformsv2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    tranformsv2.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

eval_transforms = tranformsv2.Compose([
    tranformsv2.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

if __name__ == '__main__':

    context = fl.init()

    fl.log(f"Flautim inicializado!!!")
    parquet_file = "./data/dataset_sypak_cropped.parquet"

    df = pd.read_parquet(parquet_file)
    df = dirichlet_partition(df, num_clients=1, alpha=0.5)[0]
    dataset = sypak_dataset.SypakDataset(df, train_transforms=train_transform, eval_transforms=eval_transforms)

    model = r50_model.ResNet50Classifier(context, num_classes=3)

    experiment = sypak_experiment.SypakExperiment(model, dataset, context)
    flm.Metrics.precision = skmetrics.precision_score
    flm.Metrics.recall = skmetrics.recall_score
    flm.Metrics.f1_score = skmetrics.f1_score
    experiment.run(metrics = {'ACCURACY': flm.Metrics.accuracy, 'PRECISION': flm.Metrics.precision, 'RECALL': flm.Metrics.recall, 'F1_SCORE': flm.Metrics.f1_score})
