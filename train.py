import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.lightning import LightningSystem
from src.models.cnn import ChinatownModel


def main():
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
                             train_data='train_subset.tsv',
                             val_data='val_subset.tsv',
                             test_data=None,
                             batch_size=2)

    trainer = Trainer(
        log_save_interval=1,
        row_log_interval=1000,
        # gpus=1,
        fast_dev_run=True,
        checkpoint_callback=checkpoint_callback
    )

    trainer.fit(system)


if __name__ == "__main__":
    main()
