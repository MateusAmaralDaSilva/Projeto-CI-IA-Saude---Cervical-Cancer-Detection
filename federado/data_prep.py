import pandas as pd
import numpy as np
from collections import defaultdict

def dirichlet_partition(df, num_clients=1,alpha=0.5):
    clients = defaultdict(list)
    classes = df['label'].unique()

    for cls in classes:
        cls_indices = df[df['label'] == cls].index.tolist()
        np.random.shuffle(cls_indices)

        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]

        split_indices = np.split(cls_indices, proportions)

        for i, idx in enumerate(split_indices):
            clients[i].extend(idx)
    
    partitions = []
    for i in range(num_clients):
        partition = df.loc[clients[i]].reset_index(drop=True)
        partitions.append(partition)
    return partitions
