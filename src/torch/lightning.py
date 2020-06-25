from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.classification import (F1, Accuracy,
                                                      Precision, Recall)
from torch.utils.data import DataLoader, Sampler


class LightningSystem(pl.LightningModule):

    def __init__(self,
                 model: torch.nn.Module,
                 num_classes: int,
                 train_dataset: torch.utils.data.Dataset,
                 val_dataset: torch.utils.data.Dataset = None,
                 batch_size: int = 16,
                 shuffle: bool = False,
                 num_workers: int = 1,
                 collate_fn=None,
                 train_sampler: Optional[Sampler] = None,
                 val_sampler: Optional[Sampler] = None,
                 test_sampler: Optional[Sampler] = None):
        pl.LightningModule.__init__(self)
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.criterion = torch.nn.modules.loss.BCEWithLogitsLoss()
        #self.criterion = torch.nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.Adam(params=model.parameters())
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        self.train_metrics = [Accuracy(num_classes=num_classes)]
        self.val_metrics = [
            Accuracy(num_classes=num_classes),
            F1(),
            Precision(),
            Recall()]

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.collate_fn = collate_fn
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler

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

        val_metrics = self.calc_metrics(y_hat, labels, self.val_metrics, 'val')

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {**{'val_loss': avg_loss}, **val_metrics}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(params=model.parameters())

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            self.train_dataset,
            collate_fn=self.collate_fn,
            sampler=self.train_sampler,
            batch_size=self.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.collate_fn,
            sampler=self.val_sampler,
            batch_size=self.batch_size)

    # @pl.data_loader
    # def test_dataloader(self):
    #     return DataLoader(
    #         self.dataset,
    #         sampler=self.test_sampler,
    #         batch_size=self.batch_size)def calc_metrics(y_hat, labels, metrics, prefix):
