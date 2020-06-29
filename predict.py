import torch
from pytorch_lightning import Trainer

from src.lightning import LightningSystem
from src.models.cnn import ChinatownModel

if __name__ == "__main__":
    path = '/root/epoch_41.ckpt'
    checkpoint = torch.load(path)

    d = {'.'.join(k.split('.')[1:]): v for k,
         v in checkpoint['state_dict'].items()}

    model = ChinatownModel()
    model.load_state_dict(d)

    system = LightningSystem(model=model,
                             train_data='/root/train_subs.tsv',
                             val_data='/root/val_subs.tsv',
                             test_data='/root/val_subs.tsv',
                             batch_size=128)

    trainer = Trainer(gpus=1)

    trainer.test(system)
