import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.lr_scheduler import StepLR

from src.dataset import PaddingCollateFn
from src.torch.models.cnn import ChinatownModel
from src.torch.lightning import LightningSystem
from src.transforms import Lower, OneHotCharacters, Tokenize

if __name__ == "__main__":
    transforms = [Lower(), Tokenize(), OneHotCharacters()]

    # early_stop_callback = EarlyStopping(
    #     monitor='val_Accuracy',
    #     min_delta=0.00,
    #     patience=3,
    #     verbose=False,
    #     mode='max'
    # )
    
    # model = RNN(input_size=71,
    #             rnn_hidden_size=128,
    #             rnn_num_layers=2,
    #             fc_size=64,
    #             output_size=1)

    model = ChinatownModel()

    loss = torch.nn.BCELoss

    optimizer, optimizer_args = torch.optim.SGD, {'lr': 0.01, 'momentum': 0.9}

    scheduler, scheduler_args = StepLR, {'step_size': 5, 'gamma': 0.5}

    system = LightningSystem(model=model,
                             train_data_path='/root/train_subs1.tsv',
                             val_data_path='/root/val_subs1.tsv',
                             loss=loss,
                             optimizer=optimizer,
                             optimizer_args=optimizer_args,
                             scheduler=scheduler,
                             scheduler_args=scheduler_args,
                             num_workers=1,
                             transforms=transforms,
                             num_classes=2,
                             batch_size=128,
                             collate_fn=PaddingCollateFn())

    trainer = Trainer(
        # log_save_interval=1000,
        # row_log_interval=1000,
        # val_check_interval=1000,
        #limit_val_batches=0.1,
        # distributed_backend='ddp',
        #gpus=1,
        fast_dev_run=True,
        # early_stop_callback=early_stop_callback,
        # precision=16,
        # auto_scale_batch_size=True
    )

    trainer.fit(system)
