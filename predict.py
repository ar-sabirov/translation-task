import click
import torch
from pytorch_lightning import Trainer

from src.lightning import LightningSystem
from src.models.cnn import ChinatownModel


@click.command()
@click.option('--test-path', type=str)
@click.option('--checkpoint-path', type=str)
def main(**params):
    path = params['checkpoint_path']
    checkpoint = torch.load(path)

    d = {'.'.join(k.split('.')[1:]): v for k,
         v in checkpoint['state_dict'].items()}

    model = ChinatownModel()
    model.load_state_dict(d)

    system = LightningSystem(model=model,
                             train_data='train_subset.tsv',
                             val_data='val_subset.tsv',
                             test_data=params['test_path'],
                             batch_size=128)

    trainer = Trainer(
        #gpus=1,
        fast_dev_run=True
    )

    trainer.test(system)


if __name__ == "__main__":
    main()
