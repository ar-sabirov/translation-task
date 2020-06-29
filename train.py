import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.lightning import LightningSystem
from src.models.cnn import ChinatownModel

if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='val_loss',
        mode='min',
        period=1,
        save_top_k=3,
        save_weights_only=True
    )
    
    model = ChinatownModel()

    system = LightningSystem(model=model,
                             train_data='/root/train_subs.tsv',
                             val_data='/root/val_subs.tsv',
                             test_data='/root/test_data.tsv',
                             batch_size=128)

    trainer = Trainer(
        log_save_interval=1,
        row_log_interval=1000,
        gpus=1,
        fast_dev_run=True,
        checkpoint_callback=checkpoint_callback
    )

    trainer.fit(system)
