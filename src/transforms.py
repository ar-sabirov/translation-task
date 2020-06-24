import nltk
from nltk.tokenize import RegexpTokenizer

import torch


class Tokenize(object):

    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')

    def __call__(self, sample):
        ru_name, eng_name, label = sample['ru_name'], sample['eng_name'], sample['label']

        ru_name = ' '.join(self.tokenizer.tokenize(ru_name))

        eng_name = ' '.join(self.tokenizer.tokenize(eng_name))

        return {'ru_name': ru_name,
                'eng_name': eng_name,
                'label': label}


class OneHotCharacters(object):
    def __init__(self):
        self.vocab = _get_vocab()

    def __call__(self, sample):
        ru_name, eng_name, label = sample['ru_name'], sample['eng_name'], sample['label']

        ru_oh = _inputTensor(ru_name, self.vocab)
        eng_oh = _inputTensor(eng_name, self.vocab)

        return {'ru_name': ru_oh,
                'eng_name': eng_oh,
                'label': sample['label']}


def _inputTensor(line, vocab):
    tensor = torch.zeros(len(line), len(vocab))
    for li in range(len(line)):
        letter = line[li]
        try:
            pos = vocab[letter]
        except KeyError:
            pos = vocab[' ']
        tensor[li][pos] = 1
    return tensor


def _get_vocab():

    def get_chars(lower, upper):
        l = ord(lower)
        u = ord(upper)
        return ''.join([chr(i) for i in range(l, u+1)])

    bounds = [('А', 'Я'), ('а', 'я'), ('A', 'Z'), ('a', 'z'), ('0', '9')]

    letters = [' ёЁ_'] + [get_chars(a, b) for a, b in bounds]

    return {a: i for i, a in enumerate(''.join(letters))}

if __name__ == "__main__":
    # s = {'ru_name': 'товарищество с ограниченной ответственностью 123 производственно-коммерческая фирма "корн+" ltd',
    #     'eng_name': 'Omg wtf',
    #     'label': 1}
    
    # t = Tokenize()
    
    # a = t(s)
    
    # oh = OneHotCharacters()
    
    # oh(a)
    v = _get_vocab()