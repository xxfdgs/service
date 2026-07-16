"""
@Name:  NTX.py
@Auth:  rongxing
@Date:  2023/2/20-上午10:17
@IDE:   PyCharm
@PROJECT_NAME:   $ {PROJECT_NAME}
"""
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
import torch
import numpy as np

class NTX_losses_module(torch.nn.Module):
    def __init__(self,  device,batch_size, temperature, use_cosine_similarity):
        super(NTX_losses_module, self).__init__()
        self.batch_size = cfg.train.batch_size
        self.temperature = cfg.model.temperature
        self.device = torch.device(cfg.accelerator, cfg.gpu_serial) ###device cfg.accelerator  #'cuda', 0
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(cfg.model.use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        # print('l_pos',l_pos)
        # print('r_pos', r_pos)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

@register_loss('NTX_losses')
def NTX_losses(pred, true):
    if cfg.model.loss_fun == 'NTX':
        NTX_loss = NTX_losses_module(torch.device(cfg.accelerator, cfg.gpu_serial),cfg.train.batch_size,cfg.model.temperature,
                                     cfg.model.use_cosine_similarity)
        loss = NTX_loss(pred, true)
        return loss,pred
        # return loss, pred
