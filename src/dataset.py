from typing import Optional

import pandas as pd

import torch
from torch.utils.data import Dataset


class CompanyDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 transform=None,
                 test=False,
                 nrows: Optional[int] = None):
        self.df = pd.read_csv(data_path, sep='\t', index_col=0, nrows=nrows)
        self.transform = transform
        self.test = test

        if not test:
            pos_weight = len(self.df.answer) / self.df.answer.sum()
            weights = (self.df.answer * pos_weight) + ~self.df.answer
            self.weights = torch.from_numpy(weights.values)
            self.n_pos_samples = int(self.df.answer.sum())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        ru_name, eng_name = row['ru_name'], row['eng_name']

        sample = {'ru_name': ru_name,
                  'eng_name': eng_name}

        if not self.test:
            label = row['answer']
            sample['label'] = float(label)

        if self.transform:
            sample = self.transform(sample)

        return sample


class PaddingCollateFn:
    def __init__(self, pad_seq_len: int):
        self.pad_seq_len = pad_seq_len

    def __call__(self, batch):

        ru_names = [item['ru_name'] for item in batch]
        en_names = [item['eng_name'] for item in batch]

        if 'label' in batch[0].keys():
            labels = [item['label'] for item in batch]
            labels = torch.tensor(labels, dtype=torch.float)
        else:
            labels = None

        ru_names = self.pad_batch_with_zeros(ru_names, self.pad_seq_len)
        en_names = self.pad_batch_with_zeros(en_names, self.pad_seq_len)

        return [[ru_names, en_names], labels]

    @staticmethod
    def pad_batch_with_zeros(l, max_len=None):
        if not max_len:
            max_len = max([len(i) for i in l])

        batch_size = len(l)
        sample_size = l[0].size()[1]
        n_channels = 1

        out = torch.zeros((batch_size, n_channels, max_len, sample_size))

        for i, t in enumerate(l):
            seq_len = l[i].shape[0]
            seq_len = min(seq_len, max_len)
            out[i, :, :seq_len, :] = l[i][:seq_len]

        return out
