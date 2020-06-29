from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.torch.lightning import LightningSystem

from src.torch.models.cnn import ChinatownModel

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
                             train_data='/Users/ar_sabirov/2-Data/kontur_test/train_subs.tsv',
                             val_data='/Users/ar_sabirov/2-Data/kontur_test/val_subs.tsv',
                             test_data='/Users/ar_sabirov/2-Data/kontur_test/test_task/test_data.tsv',
                             batch_size=2)

    trainer = Trainer(
        log_save_interval=1,
        row_log_interval=1000,
        # val_check_interval=1000,
        # limit_val_batches=0.1,
        # distributed_backend='ddp',
        # gpus=1 ,
        fast_dev_run=True,
        # early_stop_callback=early_stop_callback,
        # precision=16,
        # auto_scale_batch_size=True
        # checkpoint_callback=checkpoint_callback
    )

    trainer.fit(system)
