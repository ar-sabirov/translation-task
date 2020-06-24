import pytorch_lightning as pl

import torch
from typing import Optional

from torch.utils.data import DataLoader, Sampler


class LightningSystem(pl.LightningModule):

    def __init__(self, 
                 model: torch.nn.Module,
                 dataset: torch.utils.data.Dataset,
                 batch_size: int = 64,
                 criterion: Optional[torch.nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 train_sampler: Optional[Sampler] = None,
                 val_sampler: Optional[Sampler] = None,
                 test_sampler: Optional[Sampler] = None):
        pl.LightningModule.__init__(self)
        self.model = model
        self.dataset = dataset
        
        self.criterion = torch.nn.modules.loss.BCELoss()
        #self.optimizer = optimizer if optimizer else torch.optim.Adam(params=model.parameters())
        self.scheduler = scheduler
        
        self.batch_size = batch_size
        
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler
        
    def forward(self, batch) -> torch.Tensor:
        return self.model.forward(batch['ru_name'], batch['eng_name'])
    
    def training_step(self,
                      batch,
                      batch_idx: int):
        # REQUIRED
        y_hat = self.forward(batch).squeeze()
        loss = self.criterion(y_hat, batch['label'])

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch).squeeze()
        loss = self.criterion(y_hat, batch['label'])
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
    
    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            self.dataset,
            collate_fn=PaddingCollateFn(),
            sampler=self.train_sampler,
            batch_size=self.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            collate_fn=PaddingCollateFn(),
            sampler=self.val_sampler,
            batch_size=self.batch_size)

    # @pl.data_loader
    # def test_dataloader(self):
    #     return DataLoader(
    #         self.dataset,
    #         sampler=self.test_sampler,
    #         batch_size=self.batch_size)
    
    
class PaddingCollateFn:    
    def __call__(self, batch):
        
        def pad_batch_with_zeros(l):
            max_len = max([len(i) for i in l])
            batch_size = len(l)
            sample_size = l[0].size()[1]
                
            out = torch.zeros((batch_size, max_len, sample_size))
                
            for i, t in enumerate(l):
                out[i, :l[i].size()[0], :] = l[i]
                
            return out
        
        ru_names = [item['ru_name'] for item in batch]
        en_names = [item['eng_name'] for item in batch]
        labels = [item['label'] for item in batch]
        
        ru_names = pad_batch_with_zeros(ru_names)
        en_names = pad_batch_with_zeros(en_names)
        labels = torch.tensor(labels, dtype=torch.float)
        
        items = {'ru_name': ru_names,
                  'eng_name': en_names,
                  'label': labels}

        return items