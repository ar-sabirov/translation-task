from typing import Optional

import pandas as pd

import torch
from torch.utils.data import Dataset


class CompanyDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 transform=None,
                 nrows: Optional[int] = None):
        self.df = pd.read_csv(data_path, sep='\t', index_col=0, nrows=nrows)
        self.transform = transform
        self.groups = self.df['eng_name']

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


class PaddingCollateFn:
    def __call__(self, batch):
        
        def min_power_2(x: int) -> int:
            x = int(x)
            return 1 << (x-1).bit_length()

        def pad_batch_with_zeros(l):
            max_len = min_power_2(max([len(i) for i in l]))
            batch_size = len(l)
            sample_size = l[0].size()[1]

            out = torch.zeros((batch_size, max_len, sample_size))

            for i, t in enumerate(l):
                out[i, :l[i].size()[0], :] = l[i]

            return out

        ru_names = [item['ru_name'] for item in batch]
        en_names = [item['eng_name'] for item in batch]
        labels = [item['label'] for item in batch]

        ru_names = pad_batch_with_zeros(ru_names)
        en_names = pad_batch_with_zeros(en_names)
        labels = torch.tensor(labels, dtype=torch.long)

        return [[ru_names, en_names], labels]
