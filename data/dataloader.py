# _*_ coding : utf-8 _*_
# @Time : 2023/7/5 10:35
# @Author : Black
# @File : dataloader
# @Project : BabyBeatAnalyzer

import torch
import torch.utils.data as DataLoader


class FHRDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):
        super(FHRDataLoader, self).__init__(dataset, batch_size, shuffle, sampler,
                                            batch_sampler, num_workers, collate_fn,
                                            pin_memory, drop_last, timeout,
                                            worker_init_fn, multiprocessing_context)
