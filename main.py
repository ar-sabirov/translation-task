from pytorch_lightning import Trainer
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose

from src.dataset import CompanyDataset, PaddingCollateFn
from src.models import RNN
from src.torch.lightning import LightningSystem
from src.transforms import OneHotCharacters, Tokenize

if __name__ == "__main__":
    trf = [Tokenize(), OneHotCharacters()]

    ds = CompanyDataset(data_path='/Users/ar_sabirov/2-Data/kontur_test/test_task/train_data.tsv',
                        transform=Compose(trf),
                        nrows=10000)

    n_splits = 1
    train_size = 0.8

    gs_split = GroupShuffleSplit(n_splits=n_splits, train_size=train_size)

    for train_indicies, val_indicies in gs_split.split(ds, groups=ds.groups):

        train_sampler = SubsetRandomSampler(train_indicies)
        val_sampler = SubsetRandomSampler(val_indicies)

        model = RNN()

        system = LightningSystem(model=model,
                                 train_dataset=ds,
                                 val_dataset=ds,
                                 num_classes=1,
                                 batch_size=1024,
                                 collate_fn=PaddingCollateFn(),
                                 train_sampler=train_sampler,
                                 val_sampler=val_sampler)

        trainer = Trainer(log_save_interval=10,
                          #precision=16,
                          #auto_scale_batch_size=True
                          )

        trainer.fit(system)
