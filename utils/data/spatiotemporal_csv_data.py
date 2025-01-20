import argparse
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import utils.data.functions
import scipy.sparse as sp
import torch

class SpatioTemporalCSVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_workers: int = 8,
        **kwargs
    ):
        super(SpatioTemporalCSVDataModule, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.batch_size = self.hparams.batch_size
        self.seq_len = self.hparams.seq_len
        self.pre_len = self.hparams.pre_len
        self.split_ratio = self.hparams.split_ratio
        self.normalize = self.hparams.normalize
        self._feat_path = self.hparams.feat_path
        self._feat = utils.data.functions.load_features(self._feat_path)
        self._feat_max_val = np.max(self._feat)
        self.view1 = sp.load_npz(self.hparams.view1Path)
        self.view2 = sp.load_npz(self.hparams.view2Path)
        self.v1_indices = torch.load(self.hparams.v1_indicesPath)
        self.v2_indices = torch.load(self.hparams.v2_indicesPath)
        self.num_workers = num_workers


    def setup(self, stage: str = None):
        (
            self.train_dataset,
            self.val_dataset,
        ) = utils.data.functions.generate_torch_datasets(
            self._feat,
            self.seq_len,
            self.pre_len,
            split_ratio=self.split_ratio,
            normalize=self.normalize,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset),num_workers=self.num_workers,shuffle=True, drop_last=True)

    @property
    def feat_max_val(self):
        return self._feat_max_val

    @property
    def adj(self):
        return self.view1
    @property
    def view_1(self):
        view = utils.data.functions.sparse_mx_to_torch_sparse_tensor(utils.data.functions.normalize_adj(self.view1))
        return view

    @property
    def view_2(self):
        view = utils.data.functions.sparse_mx_to_torch_sparse_tensor(utils.data.functions.normalize_adj(self.view2))
        return view
    @property
    def indices_v1(self):
        return self.v1_indices

    @property
    def indices_v2(self):
        return self.v2_indices