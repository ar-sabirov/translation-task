import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset import PaddingCollateFn
from src.torch.models.cnn import ChinatownModel
from src.torch.lightning import LightningSystem
from src.transforms import Lower, OneHotCharacters, Tokenize

from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    transforms = [Lower(), Tokenize(), OneHotCharacters()]

    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='val_loss',
        mode='min',
        period=5
    )

    model = ChinatownModel()

    loss = torch.nn.BCELoss

    optimizer, optimizer_args = torch.optim.SGD, {'lr': 0.005, 'momentum': 0.9}

    scheduler, scheduler_args = ReduceLROnPlateau, {'factor': 0.5, 'verbose': True, 'cooldown': 1}

    system = LightningSystem(model=model,
                             train_data_path='/root/train_subs.tsv',
                             val_data_path='/root/val_subs.tsv',
                             loss=loss,
                             optimizer=optimizer,
                             optimizer_args=optimizer_args,
                             scheduler=scheduler,
                             scheduler_args=scheduler_args,
                             num_workers=1,
                             transforms=transforms,
                             num_classes=2,
                             batch_size=128,
                             collate_fn=PaddingCollateFn(150))

    trainer = Trainer(
        log_save_interval=1000,
        row_log_interval=1000,
        val_check_interval=1000,
        #limit_val_batches=0.1,
        # distributed_backend='ddp',
        gpus=1,
        fast_dev_run=True,
        # early_stop_callback=early_stop_callback,
        # precision=16,
        # auto_scale_batch_size=True
        checkpoint_callback=checkpoint_callback
    )

    trainer.fit(system)
