from src.torch.lightning import LightningSystem
from pytorch_lightning import Trainer

from src.torch.models.cnn import ChinatownModel
import torch

if __name__ == "__main__":
    path = '/Users/ar_sabirov/1-Code/translation-task/lightning_logs/version_0/checkpoints/epoch=1.ckpt'
    checkpoint = torch.load(path)
    
    d = {'.'.join(k.split('.')[1:]) : v for k,v in checkpoint['state_dict'].items()}
    
    model = ChinatownModel()
    model.load_state_dict(d)

    system = LightningSystem(model=model,
                             train_data='/Users/ar_sabirov/2-Data/kontur_test/train_subs.tsv',
                             val_data='/Users/ar_sabirov/2-Data/kontur_test/val_subs.tsv',
                             test_data='/Users/ar_sabirov/2-Data/kontur_test/test_task/test_data.tsv',
                             batch_size=2)

    trainer = Trainer(fast_dev_run=True)
    
    trainer.test(system)