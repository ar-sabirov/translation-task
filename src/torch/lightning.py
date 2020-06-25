from typing import Optional

import pytorch_lightning as pl
import torch
# from pytorch_lightning.metrics.classification import (F1, Accuracy, Precision,
#                                                       Recall)
#from pytorch_lightning.metrics import Accuracy
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose

from src.dataset import CompanyDataset


class LightningSystem(pl.LightningModule):

    def __init__(self,
                 model: torch.nn.Module,
                 num_classes: int,
                 num_workers: int,
                 train_data_path: str,
                 val_data_path: str,
                 transforms=[],
                 batch_size: int = 16,
                 shuffle: bool = False,
                 collate_fn=None):
        pl.LightningModule.__init__(self)
        self.model = model
        self.transforms = Compose(transforms)

        self.criterion = torch.nn.modules.loss.BCEWithLogitsLoss()
        #self.criterion = torch.nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.Adam(params=model.parameters())
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        #self.train_metrics = [Accuracy(num_classes=num_classes)]
        self.train_metrics = []
        # self.val_metrics = [
        #     Accuracy(num_classes=num_classes),
        #     F1(),
        #     Precision(),
        #     Recall()]
        self.val_metrics = []
        
        self.num_workers = num_workers

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.collate_fn = collate_fn
        
        self.train_dataset = CompanyDataset(data_path=train_data_path,
                                            transform=self.transforms)
        
        self.val_dataset = CompanyDataset(data_path=val_data_path,
                                          transform=self.transforms)

    @staticmethod
    def calc_metrics(y_hat, labels, metrics, prefix):
        return {f'{prefix}_{metric._get_name()}': metric(y_hat, labels) for metric in metrics}

    def forward(self, inputs) -> torch.Tensor:
        return self.model.forward(inputs)

    def training_step(self,
                      batch,
                      batch_idx: int):
        # REQUIRED
        inputs, labels = batch
        y_hat = self.forward(inputs).squeeze()
        loss = self.criterion(y_hat, labels)

        train_metrics = self.calc_metrics(
            y_hat, labels, self.train_metrics, 'train')

        tensorboard_logs = {**{'train_loss': loss}, **train_metrics}

        return {'loss': loss,
                'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        y_hat = self.forward(inputs).squeeze()
        loss = self.criterion(y_hat, labels)
        
        return {'val_loss': loss, 'labels': labels, 'preds': y_hat}

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x['labels'] for x in outputs])
        y_hat = torch.cat([x['preds'] for x in outputs])

        #val_metrics = self.calc_metrics(y_hat, labels, self.val_metrics, 'val')
        
        preds = (y_hat > 0.5).to(dtype=torch.float)
        f1_score = f1_score(labels, preds)

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'f1': f1_score}
        return {'val_loss': avg_loss,
                'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters())

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size)

    # @pl.data_loader
    # def test_dataloader(self):
    #     return DataLoader(
    #         self.dataset,
    #         sampler=self.test_sampler,
    #         batch_size=self.batch_size)def calc_metrics(y_hat, labels, metrics, prefix):

def f1_score(y_true, y_pred) -> torch.Tensor:
    tp = (y_true * y_pred).sum().to(torch.float)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return f1