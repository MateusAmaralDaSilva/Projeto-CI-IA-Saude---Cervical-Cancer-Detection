import torch
from pathlib import Path
import matplotlib.pyplot as plt
from flautim.pytorch.Dataset import Dataset
import flautim as fl
from torch.utils.data import random_split, DataLoader
import numpy as np
import copy

class SypakDataset(Dataset):
    
    def __init__(self, df):
        super(SypakDataset, self).__init__(name = "Sypak")
        self.df = df
        self.X = np.stack(df.drop('label', axis=1).values).astype(np.uint8)
        self.y = df['label'].astype('category').cat.codes.values

        torch.manual_seed(42)
        total_size = len(self.df)
        eval_size = int(total_size * 0.2)
        train_size = total_size - eval_size
        self.train_subset, self.eval_subset = random_split(
            range(total_size), [train_size, eval_size]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).reshape(3, 224, 224) / 255.0
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return {"image": x, "label": y}
    
    def dataloader(self, validation=False, batch_size=32, num_workers=1):
        indices = self.eval_subset if validation else self.train_subset
        subset = torch.utils.data.Subset(self, indices)
        return DataLoader(subset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)