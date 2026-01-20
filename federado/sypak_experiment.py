from flautim.pytorch.federated.Experiment import Experiment
import flautim as fl
import flautim.metrics as flm
import numpy as np
import torch
import time
import sklearn.metrics as skmetrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#from sklearn.linear_model import LogisticRegression
from statsmodels.miscmodels.ordinal_model import OrderedModel
from matplotlib import pyplot as plt
from pathlib import Path

plot_path = './plots/'

def count_files(directory: str) -> int:
    return len([f for f in Path(directory).iterdir() if f.is_file()])

def plot_embeddings(embeddings, labels):

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Class Label')
    plt.title('Embeddings Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.savefig(f'{plot_path}embeddings{count_files(plot_path)}.png')
    plt.close()

def ordinal_regression_validation(embeddings, labels):
    model = OrderedModel(
        labels,
        embeddings,
        distr="logit"
    )

    reg = model.fit(method="bfgs", disp=False)

    probs = reg.predict(embeddings)

    y_pred = probs.argmax(axis=1)
    
    error = (labels - y_pred).astype(float)
    ss_res = np.sum((labels - y_pred) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    
    mse = np.mean(error ** 2)
    mae = np.mean(np.abs(error))
    var_error = np.var(error)
    r2_score = 1 - (ss_res / ss_tot)
    return mse, mae, var_error, r2_score #reg

def clustering_validation(embeddings, labels):
    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(embeddings)

    nmi = skmetrics.normalized_mutual_info_score(labels, cluster_labels)
    ari = skmetrics.adjusted_rand_score(labels, cluster_labels)
    silhouette = skmetrics.silhouette_score(embeddings, cluster_labels)
    squared_error = skmetrics.mean_squared_error(embeddings, cluster_labels)
    
    return nmi, ari, silhouette, squared_error
    
class SypakExperiment(Experiment):
    def __init__(self, model, dataset, context, **kwargs):
        super(SypakExperiment, self).__init__(model, dataset, context, **kwargs)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = kwargs.get('lr', 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.epochs = kwargs.get('epochs', 10)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def training_loop(self, data_loader):
        self.dataset.train = True
        self.model.to(self.device)
        self.model.train()

        running_loss = 0.0

        yhat, y_real, all_emb = [], [], []

        for batch in data_loader:
            images, labels = batch
            self.optimizer.zero_grad()
            logits, emb = self.model(images.to(self.device), return_embeddings=True)
            loss = self.criterion(logits, labels.to(self.device))
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            all_emb.append(emb.detach().cpu())
            yhat.append(predicted.detach().cpu())
            y_real.append(labels.detach().cpu())

        accuracy = flm.Metrics.accuracy(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        precision = skmetrics.precision_score(torch.cat(y_real).numpy(), torch.cat(yhat).numpy(), average='weighted', zero_division=0)
        recall = skmetrics.recall_score(torch.cat(y_real).numpy(), torch.cat(yhat).numpy(), average='weighted', zero_division=0)
        f1_score = skmetrics.f1_score(torch.cat(y_real).numpy(), torch.cat(yhat).numpy(), average='weighted', zero_division=0)
        
        nmi, ari, silhouette, squared_error = clustering_validation(
            embeddings=torch.cat(all_emb).detach().cpu().numpy(),
            labels=torch.cat(y_real).numpy()
        )
        nmi, ari, silhouette, squared_error = float(nmi), float(ari), float(silhouette), float(squared_error)
        
        mse, mae, var_error, r2 = ordinal_regression_validation(
            embeddings=torch.cat(all_emb).detach().cpu().numpy(),
            labels=torch.cat(y_real).numpy()
        )
        
        avg_trainloss = running_loss / len(data_loader)
        
        return float(avg_trainloss), {'ACCURACY': accuracy, 'PRECISION': precision, 'RECALL': recall, 'F1_SCORE': f1_score, 'NMI': nmi, 'ARI': ari, 'SILHOUETTE': silhouette, 'SQUARED_ERROR': squared_error,'MSE': mse, 'MAE': mae, 'VAR_ERROR': var_error, 'R2': r2}

    def validation_loop(self, data_loader):
        self.dataset.train = False
        self.model.to(self.device)
        self.model.eval()

        yhat, y_real, all_emb = [], [], []

        loss = 0.0
        with torch.no_grad():
            for batch in data_loader:
                images, labels = batch
                images = images.to(self.device)
                labels =labels.to(self.device)
                logits, emb = self.model(images, return_embeddings=True)
                loss += self.criterion(logits, labels).item()
                _, predicted = torch.max(logits.data, 1)
                all_emb.append(emb.detach().cpu())
                yhat.append(predicted.detach().cpu())
                y_real.append(labels.detach().cpu())
        
        
        accuracy = flm.Metrics.accuracy(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        precision = skmetrics.precision_score(torch.cat(y_real).numpy(), torch.cat(yhat).numpy(), average='weighted', zero_division=0)
        recall = skmetrics.recall_score(torch.cat(y_real).numpy(), torch.cat(yhat).numpy(), average='weighted', zero_division=0)
        f1_score = skmetrics.f1_score(torch.cat(y_real).numpy(), torch.cat(yhat).numpy(), average='weighted', zero_division=0)
        loss = loss / len(data_loader)
        
        nmi, ari, silhouette, squared_error = clustering_validation(
            embeddings=torch.cat(all_emb).detach().cpu().numpy(),
            labels=torch.cat(y_real).numpy()
        )
        #transformando np.float64 em float32 para evitar problemas de serialização no FL
        nmi, ari, silhouette, squared_error = float(nmi), float(ari), float(silhouette), float(squared_error)
        
        
        mse, mae, var_error, r2 = ordinal_regression_validation(
            embeddings=torch.cat(all_emb).detach().cpu().numpy(),
            labels=torch.cat(y_real).numpy()
        )
        
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(torch.cat(all_emb).detach().cpu().numpy())
        plot_embeddings(embeddings_2d, torch.cat(y_real).numpy())
        
        @staticmethod
        def count_files(directory: str) -> int:
            return len([f for f in Path(directory).iterdir() if f.is_file()])
        self.save_index = self.count_files(self.save_directory)
        return float(loss), {'ACCURACY': accuracy, 'PRECISION': precision, 'RECALL': recall, 'F1_SCORE': f1_score, 'NMI': nmi, 'ARI': ari, 'SILHOUETTE': silhouette, 'SQUARED_ERROR': squared_error,'MSE': mse, 'MAE': mae, 'VAR_ERROR': var_error, 'R2': r2}