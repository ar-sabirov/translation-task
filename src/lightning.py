import pytorch_lightning as pl
import torch
# from pytorch_lightning.metrics.classification import (F1, Accuracy, Precision,
#                                                       Recall)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import Compose

from src.dataset import CompanyDataset, PaddingCollateFn
from src.transforms import Lower, OneHotCharacters, Tokenize
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.util import save_predictions


class LightningSystem(pl.LightningModule):

    def __init__(self,
                 model,
                 train_data,
                 val_data,
                 batch_size,
                 test_data = None,
                 num_classes=2,
                 num_workers=1):
        super().__init__()
        self.num_workers = num_workers
        self.train_data=train_data
        self.val_data=val_data
        self.test_data= test_data
        self.batch_size=batch_size

        self.model = model

        self.criterion = torch.nn.BCELoss()

        self.collate_fn = PaddingCollateFn(150)

        self.train_metrics = []
        self.val_metrics = []
        self.val_metrics = []
        # self.val_metrics = [
        #     Accuracy(num_classes=num_classes),
        #     F1(num_classes=num_classes),
        #     Precision(num_classes=num_classes),
        #     Recall(num_classes=num_classes)]
        transforms = Compose([Lower(), Tokenize(), OneHotCharacters()])

        self.train_dataset = CompanyDataset(data_path=self.train_data,
                                            transform=transforms)

        # self.train_sampler = WeightedRandomSampler(weights=self.train_dataset.weights,
        #                                            num_samples=2 * self.train_dataset.n_pos_samples,
        #                                            replacement=True)

        self.val_dataset = CompanyDataset(data_path=self.val_data,
                                          transform=transforms)

        if self.test_data:
            self.test_dataset = CompanyDataset(data_path=self.test_data,
                                               test=True,
                                               transform=transforms)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=0.005,
                                    momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer,
                                      factor=0.5,
                                      verbose=True,
                                      cooldown=1)
        return [optimizer], [scheduler]

    @staticmethod
    def _calc_metrics(y_hat, labels, metrics, prefix):
        return {f'{prefix}_{metric._get_name()}': metric(y_hat, labels) for metric in metrics}

    def forward(self, inputs) -> torch.Tensor:
        return self.model.forward(inputs).squeeze()

    def training_step(self,
                      batch,
                      batch_idx: int):
        inputs, labels = batch
        y_hat = self.forward(inputs)
        loss = self.criterion(y_hat, labels)

        train_metrics = self._calc_metrics(
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

        val_metrics = self._calc_metrics(
            preds, labels.to(torch.long), self.val_metrics, 'val')

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {**{'val_loss': avg_loss}, **val_metrics}
        return {'val_loss': avg_loss,
                'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        y_hat = self.forward(inputs)
        loss = self.criterion(y_hat, labels)

        return {'val_loss': loss, 'labels': labels, 'preds': y_hat}

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        y_hat = self.forward(inputs)
        preds = (y_hat > 0.5).to(dtype=torch.long)

        return {'preds': preds}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs])
        preds_arr = preds.detach().cpu().numpy()

        save_predictions(preds_arr)

        return {}

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            #sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size)

    @pl.data_loader
    def test_dataloader(self):
        assert self.test_data, 'Please provide test data for testing'
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size)
