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
        save_weights_only=True
    )
    
    path = '/root/epoch=13.ckpt'
    checkpoint = torch.load(path)
    
    d = {'.'.join(k.split('.')[1:]) : v for k,v in checkpoint['state_dict'].items()}
    
    model = ChinatownModel()
    model.load_state_dict(d)

    system = LightningSystem(model=model,
                             train_data='/root/train_subs.tsv',
                             val_data='/root/val_subs.tsv',
                             test_data=None,
                             batch_size=256)

    trainer = Trainer(
        # log_save_interval=1000,
        # row_log_interval=1000,
        gpus=-1,
        #max_epochs=30,
        logger=False,
        distributed_backend='ddp',
        #fast_dev_run=True,
        checkpoint_callback=checkpoint_callback
    )

    trainer.fit(system)
