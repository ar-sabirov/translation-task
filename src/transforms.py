from nltk.tokenize import RegexpTokenizer

import torch
from src.util import _get_vocab


class Lower:
    def __call__(self, sample):
        sample['ru_name'] = sample['ru_name'].lower()
        sample['eng_name'] = sample['eng_name'].lower()
        return sample


class Tokenize:

    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')

    def __call__(self, sample):
        sample['ru_name'] = ' '.join(
            self.tokenizer.tokenize(sample['ru_name']))
        sample['eng_name'] = ' '.join(
            self.tokenizer.tokenize(sample['eng_name']))
        return sample


class OneHotCharacters:
    def __init__(self):
        self.vocab = _get_vocab()

    def __call__(self, sample):
        sample['ru_name'] = self._inputTensor(sample['ru_name'], self.vocab)
        sample['eng_name'] = self._inputTensor(sample['eng_name'], self.vocab)
        return sample

    @staticmethod
    def _inputTensor(line, vocab):
        tensor = torch.zeros(len(line), len(vocab))
        for i, letter in enumerate(line):
            try:
                pos = vocab[letter]
            except KeyError:
                pos = vocab[' ']
            tensor[i][pos] = 1
        return tensor
