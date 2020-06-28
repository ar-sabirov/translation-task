from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.classification import (F1, Accuracy, Precision,
                                                      Recall)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import Compose

from src.dataset import CompanyDataset


class LightningSystem(pl.LightningModule):

    def __init__(self,
                 model: torch.nn.Module,
                 num_classes: int,
                 num_workers: int,
                 train_data_path: str,
                 val_data_path: str,
                 loss,
                 optimizer,
                 optimizer_args,
                 scheduler,
                 scheduler_args,
                 transforms=[],
                 batch_size: int = 16,
                 collate_fn=None):
        pl.LightningModule.__init__(self)
        self.model = model
        self.transforms = Compose(transforms)

        self.criterion = loss()
        self.optimizer = optimizer(self.parameters(), **optimizer_args)
        self.scheduler = scheduler(self.optimizer, **scheduler_args)

        self.train_metrics = []
        self.val_metrics = []
        self.val_metrics = [
            Accuracy(num_classes=num_classes),
            F1(),
            Precision(),
            Recall()]

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def prepare_data(self):
        self.train_dataset = CompanyDataset(data_path=train_data_path,
                                            transform=self.transforms)
        
        self.train_sampler = WeightedRandomSampler(weights=self.train_dataset.weights,
                                                   num_samples=2 * self.train_dataset.n_pos_samples,
                                                   replacement=True)

        self.val_dataset = CompanyDataset(data_path=val_data_path,
                                          transform=self.transforms)

        self.val_sampler = WeightedRandomSampler(weights=self.val_dataset.weights,
                                                 num_samples=2 * self.val_dataset.n_pos_samples,
                                                 replacement=True)

    @staticmethod
    def calc_metrics(y_hat, labels, metrics, prefix):
        return {f'{prefix}_{metric._get_name()}': metric(y_hat, labels) for metric in metrics}

    def forward(self, inputs) -> torch.Tensor:
        return self.model.forward(inputs).squeeze()

    def training_step(self,
                      batch,
                      batch_idx: int):
        # REQUIRED
        inputs, labels = batch
        y_hat = self.forward(inputs)
        loss = self.criterion(y_hat, labels)

        train_metrics = self.calc_metrics(
            y_hat, labels, self.train_metrics, 'train')

        tensorboard_logs = {**{'train_loss': loss}, **train_metrics}

        return {'loss': loss,
                'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        y_hat = self.forward(inputs)
        loss = self.criterion(y_hat, labels)

        return {'val_loss': loss, 'labels': labels, 'preds': y_hat}

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x['labels'] for x in outputs])
        y_hat = torch.cat([x['preds'] for x in outputs])

        preds = (y_hat > 0.5).to(dtype=torch.long)

        val_metrics = self.calc_metrics(
            preds, labels.to(torch.long), self.val_metrics, 'val')

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {**{'val_loss': avg_loss}, **val_metrics}
        return {'val_loss': avg_loss,
                'log': tensorboard_logs}

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            sampler=self.val_sampler,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size)

    # @pl.data_loader
    # def test_dataloader(self):
    #     return DataLoader(
    #         self.dataset,
    #         sampler=self.test_sampler,
    #         batch_size=self.batch_size)
