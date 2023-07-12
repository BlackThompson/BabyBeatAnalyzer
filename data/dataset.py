# _*_ coding : utf-8 _*_
# @Time : 2023/7/5 10:35
# @Author : Black
# @File : dataset
# @Project : BabyBeatAnalyzer

import torch
from torch.utils.data import Dataset
from data.utils import preprocess


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = preprocess(self.data[index])
        # data = self.data[index]
        data = torch.tensor(data, dtype=torch.float32)
        label = self.label[index]
        label = torch.tensor(label, dtype=torch.long)
        return data, label

    def __len__(self):
        return len(self.data)
