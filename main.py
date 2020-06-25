from sklearn.model_selection import GroupShuffleSplit
from torchvision.transforms import Compose

from src.data.dataset import CompanyDataset
from src.transforms import Tokenize, OneHotCharacters
from torch.utils.data.sampler import SubsetRandomSampler

from src.torch.lightning import LightningSystem

from src.models import RNN

from pytorch_lightning import Trainer

if __name__ == "__main__":    
    trf = [Tokenize(), OneHotCharacters()]
    
    ds = CompanyDataset(data_path='/Users/ar_sabirov/2-Data/kontur_test/test_task/train_data.tsv',
                        transform=Compose(trf),
                        nrows=10000)
    
    n_splits = 1
    train_size = 0.8
    
    gs_split = GroupShuffleSplit(n_splits=n_splits, train_size=train_size)
    
    train_indicies, val_indicies = list(gs_split.split(ds, groups=ds.groups))[0]
    train_sampler = SubsetRandomSampler(train_indicies)
    val_sampler = SubsetRandomSampler(val_indicies)
    
    model = RNN()
    
    system = LightningSystem(model=model,
                             dataset=ds,
                             batch_size=128,
                             train_sampler=train_sampler,
                             val_sampler=val_sampler)
    
    trainer = Trainer()
    
    trainer.fit(system)
