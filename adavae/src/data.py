import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
import os
import torch
import functools

class DataFrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, slice_orient='list'):
        self.df = df
        self.slice_orient = slice_orient

    def __getitem__(self, index) -> dict:
        if isinstance(index, slice):
            return self.df.iloc[index].to_dict(orient=self.slice_orient)
        else:
            return self.df.iloc[index].to_dict()

    def __len__(self):
        return len(self.df)


class DataFrameTextClassificationDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 x_label: str = 'text',
                 y_label: str = 'label'):
        self.x = df[x_label]
        self.length = len(self.x)

        self.y = df[y_label].astype('category')
        self.n_classes = len(self.y.cat.categories)
        self.y = self.y.cat.codes

    def __getitem__(self, index) -> dict:
        x = self.x.iloc[index]
        y = self.y.iloc[index]
        return {'x': str(x), 'y': int(y)}

    def __len__(self):
        return self.length

    @staticmethod
    def from_file(file_path: str,
                  x_label: str = 'text',
                  y_label: str = 'label'):
        df = pd.read_csv(file_path)
        return DataFrameTextClassificationDataset(df, x_label, y_label)


class ConditionalGenerationDataset(Dataset):
    def __init__(self, dl: list):
        self.x = []
        self.text_len = []
        self.y = []
        self.init_data(dl)
        self.length = len(self.x)

    def init_data(self, dl):
        for inst in dl:
            inst = inst.split('\t')
            ## label
            self.y.append(inst[0])
            self.x.append(inst[1])
            self.text_len.append(len(inst[1].split()))

    def __getitem__(self, index: int) -> dict:
        ## add BOS and EOS special token
        x = '<|endoftext|> ' + self.x[index][:-1] + ' <|endoftext|>'
        y = self.y[index]

        return {'x': str(x), 'y': int(y)}

    def __len__(self):
        return self.length

    ## call for direct input
    @staticmethod
    def from_file(file_path: str):
        with open(file_path, 'r') as f:
            dl = f.readlines()
        return ConditionalGenerationDataset(dl)

class GenerationDataset(Dataset):
    def __init__(self, dl: list):
        self.x = []
        self.text_len = []
        self.init_data(dl)
        self.length = len(self.x)

    def init_data(self, dl):
        for inst in dl:
            ## label
            self.x.append(inst)
            self.text_len.append(len(inst.split()))

    def __getitem__(self, index: int) -> dict:
        ## add BOS and EOS special token
        x = '<|endoftext|> ' + self.x[index] + ' <|endoftext|>'

        return {'x': str(x)}

    def __len__(self):
        return self.length

    ## call for direct input
    @staticmethod
    def from_file(file_path: str):
        with open(file_path, 'r') as f:
            dl = f.readlines()
        return GenerationDataset(dl)

class GLUEPretrainingDataset(Dataset):
    def __init__(self, dl: list, dataset: str):
        self.x = []
        self.text_len = []
        self.init_data(dl, dataset)
        self.length = len(self.x)

    def init_data(self, dl, dataset):
        for inst in dl:
            inst = inst.split("\t")
            if dataset == "cola":
                self.x.append(inst[3])
                self.text_len.append(len(inst[3].split()))
            elif dataset == "sst-2":
                self.x.append(inst[0])
                self.text_len.append(len(inst[0].split()))

    def __getitem__(self, index: int) -> dict:
        ## add BOS and EOS special token
        x = '<|endoftext|> ' + self.x[index] + ' <|endoftext|>'

        return {'x': str(x)}

    def __len__(self):
        return self.length

    ## call for direct input
    @staticmethod
    def from_file(file_path: str, dataset: str):
        with open(file_path, 'r') as f:
            dl = f.readlines()
        return GLUEPretrainingDataset(dl, dataset)

class DialogGenerationDataset(Dataset):
    def __init__(self, dl: list):
        self.x = []
        self.text_len = []
        self.y = []
        self.init_data(dl)
        self.length = len(self.x)

    def init_data(self, dl):
        for inst in dl:
            inst = inst.split('\t')
            ## context
            self.y.append(inst[0])
            ## response
            self.x.append(inst[1])
            self.text_len.append(len(inst[1].split()))

    def __getitem__(self, index: int) -> dict:
        ## add BOS and EOS special token
        x = '<|endoftext|> ' + self.x[index] + ' <|endoftext|>'
        y = '<|endoftext|> ' + self.y[index] + ' <|endoftext|>'

        return {'response': str(x), 'context': str(y)}

    def __len__(self):
        return self.length

    ## call for direct input
    @staticmethod
    def from_file(file_path: str):
        with open(file_path, 'r') as f:
            dl = f.readlines()
        return DialogGenerationDataset(dl)

class DictDataset(Dataset):
    def __init__(self, dl):
        self.text_len = []
        self.dl = dl
        self.length = len(self.dl)

    def __getitem__(self, index: int) -> dict:
        ## add BOS and EOS special token
        data_dict = {}
        data_dict['guid'] = self.dl[index].guid
        data_dict['text_a'] = self.dl[index].text_a
        if not self.dl[index].text_b is None:
            data_dict['text_b'] = self.dl[index].text_b
        data_dict['label'] = int(self.dl[index].label)

        return data_dict

    def __len__(self):
        return self.length

def collate_fn(samples: dict, eos_id: list, tokenizer):
    """ Creates a batch out of samples for direct input"""
    x_max_len = max(map(lambda s: len(s['x']), samples))
    # Zero pad mask
    x_mask = torch.ByteTensor([[1] * len(ss['x']) + [0] * (x_max_len - len(ss['x'])) for ss in samples])
    # tokenizer.convert_tokens_to_ids('<|startoftext|>') = 50257, endoftext 50256, use 50257 here causes errors!!
    x = torch.LongTensor([ss['x'] + eos_id * (x_max_len - len(ss['x'])) for ss in samples])


def prepare_dataset(data_dir, dataset_name, tokenizer, train_bsz, train_seq_len, val_bsz, val_seq_len, test_bsz=1,
                    test_seq_len=1024, data_type='t0', num_workers=1, make_train=True, make_val=True, make_test=False):
    loaders = []
    if make_train:
        train_dataset = ConditionalGenerationDataset.from_file('./data/yelp_polarity/train.txt')