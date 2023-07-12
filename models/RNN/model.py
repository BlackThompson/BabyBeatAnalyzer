# _*_ coding : utf-8 _*_
# @Time : 2023/7/7 14:36
# @Author : Black
# @File : model
# @Project : BabyBeatAnalyzer

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
import numpy as np
import torch.nn.functional as F


# def _get_out(out, length, batch_size):
#     out, _ = torch.nn.utils.rnn.pad_packed_sequence(sequence=out, batch_first=True)
#     fc_in = []
#     for index in range(batch_size):
#         fc_in.append(out[index][length[index] - 1])
#     return torch.stack(fc_in)


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers, type='GRU'):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        # 定义RNN层
        if type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # 定义激活函数
        self.relu = nn.ReLU(inplace=True)

        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, packed_input):
        # packed_input: [batch_size, seq_len, input_size]
        packed_output, _ = self.rnn(packed_input)

        # output: [batch_size, seq_len, hidden_size]
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # last_output: [batch_size, hidden_size]
        last_output = self.get_last_output(output, lengths)

        fc_in = self.relu(last_output)
        fc_out = self.fc(fc_in)
        return fc_out

    def get_last_output(self, output, lengths):
        batch_size = output.size(0)
        last_output = torch.zeros(batch_size, self.hidden_size, device=output.device)
        for i in range(batch_size):
            last_output[i] = output[i, lengths[i] - 1]
        return last_output
