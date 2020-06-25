from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from src.dataset import PaddingCollateFn
from src.models import RNN
from src.torch.lightning import LightningSystem
from src.transforms import OneHotCharacters, Tokenize

if __name__ == "__main__":
    transforms = [Tokenize(), OneHotCharacters()]

    # early_stop_callback = EarlyStopping(
    #     monitor='val_Accuracy',
    #     min_delta=0.00,
    #     patience=3,
    #     verbose=False,
    #     mode='max'
    # )

    model = RNN()

    system = LightningSystem(model=model,
                             train_data_path='/Users/ar_sabirov/2-Data/kontur_test/train_subs.tsv',
                             val_data_path='/Users/ar_sabirov/2-Data/kontur_test/val_subs.tsv',
                             num_workers=0,
                             transforms=transforms,
                             num_classes=2,
                             batch_size=128,
                             collate_fn=PaddingCollateFn())

    trainer = Trainer(log_save_interval=10,
                      # gpus=[0],
                      # fast_dev_run=True,
                      # early_stop_callback=early_stop_callback,
                      # precision=16,
                      # auto_scale_batch_size=True
                      )

    trainer.fit(system)
