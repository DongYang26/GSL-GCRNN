import argparse
import torch.nn as nn
from models.modules.View_Estimator import View_Estimator
from models.modules.GcRNNCell import GcRNNCell


class GSLGCRNN(nn.Module):
    def __init__(self, data, arg, hidden_dim, **kwargs):
        super(GSLGCRNN, self).__init__()
        self._input_dim = data.adj.shape[0]
        self.data = data
        self._hidden_dim = hidden_dim
        self.gcrnn = GcRNNCell(self._input_dim, self._hidden_dim)
        self.view = View_Estimator(arg.batch_size, self._hidden_dim, com_lambda_v1=arg.com_lambda_v1, com_lambda_v2=arg.com_lambda_v2, dropout=arg.ve_dropout)

    def get_view(self, data):
        new_v1, new_v2 = self.view(self.data, self._input_dim, data)
        return new_v1, new_v2

    def get_outputs(self, inputs, hidden_state, i, v1):  # input: v, feature, hidden   Output: embedding
        output_v1 = self.gcrnn(inputs[:, i, :], hidden_state, v1)
        return output_v1

    def get_preditctions(self, out):
        preditctions = self.gcrnn.projection(out)
        return preditctions

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
