from torch.utils.data import Dataset
import pandas as pd
from typing import Optional

class CompanyDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 transform = None,
                 nrows: Optional[int] = None):
        self.df = pd.read_csv(data_path, sep='\t', index_col=0, nrows=nrows)
        self.transform = transform
        self.groups = self.df['ru_name']

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        sample = self.df.iloc[idx]
        
        sample = {'idx': idx,
                  'ru_name': sample['ru_name'],
                  'eng_name': sample['eng_name'],
                  'label': float(sample['answer'])}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    