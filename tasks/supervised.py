import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils.metrics
import utils.losses
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SupervisedForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss="mse",
        feat_max_val: float = 1.0,
        **kwargs
    ):
        super(SupervisedForecastTask, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self._loss = loss
        self.feat_max_val = feat_max_val

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size, seq_len, num_nodes = x.size()
        assert self.model._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self.model._hidden_dim).type_as(x)

        output = None
        for i in range(seq_len):
            new_v1, new_v2 = self.model.get_view(x[:, i, :])
            alpha = 0.5
            view = alpha * new_v1 + (1 - alpha) * new_v2
            output = self.model.get_outputs(x, hidden_state, i, view)
        dec_intput = output.reshape((batch_size, num_nodes, self.model._hidden_dim)).to(device)
        hidden = self.model.get_preditctions(dec_intput)
        hidden = hidden.reshape((-1, hidden.size(2)))
        predictions = hidden
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        return predictions

    def shared_step(self, batch, batch_idx):
        x, y = batch
        num_nodes = x.size(2)
        predictions = self(x)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        y = y.reshape((-1, y.size(2)))
        return predictions, y

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch, batch_idx,optimizer_idx):
        if optimizer_idx == 0:
            prediction_0, y = self.shared_step(batch, batch_idx)
            loss_0 = self.loss(prediction_0, y)
            self.log("train_loss_1", loss_0)
            return loss_0

        elif optimizer_idx == 1:
            prediction_1, y = self.shared_step(batch, batch_idx)
            loss_1 = self.loss(prediction_1, y)
            self.log("train_loss_2",loss_1)
            return loss_1

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        predictions = predictions * self.feat_max_val
        predictions = torch.round(torch.abs(predictions))
        y = y * self.feat_max_val
        loss = self.loss(predictions, y)
        accuracy = utils.metrics.accuracy(predictions, y)
        r2 = utils.metrics.r2(predictions, y)
        explained_variance = utils.metrics.explained_variance(predictions, y)
        mape = utils.metrics.MAPE(y, predictions)
        mae = utils.metrics.MAE(y, predictions)
        rmse = utils.metrics.RMSE(y, predictions)
        masked_mape = utils.metrics.MASKED_MAPE(y, predictions)
        metrics = {
            "val_loss": loss,
            "RMSE": rmse,
            "MAE": mae,
            "accuracy": accuracy,
            "R2": r2,
            "ExplainedVar": explained_variance,
            "MAPE": mape,
            "MASKED_MAPE": masked_mape
        }
        self.log_dict(metrics)
        return predictions.reshape(batch[1].size()), y.reshape(batch[1].size())

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer_view = torch.optim.Adam(self.model.view.parameters(), lr=self.hparams.ve_lr, weight_decay=self.hparams.ve_weight_decay, )
        optimizer_gcrnn = torch.optim.Adam(self.model.gcrnn.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay,)
        
        lr_scheduler_view = {
            "scheduler": ReduceLROnPlateau(optimizer_view, factor=0.5, min_lr=1e-5),
            "monitor": "train_loss_1",
            "interval": "epoch",
            "frequency": 500
        }
        lr_scheduler_gcrnn = {
            "scheduler": ReduceLROnPlateau(optimizer_gcrnn, factor=0.5, min_lr=1e-5),
            "monitor": "train_loss_2",
            "interval": "epoch",
            "frequency": 500
        }
        return [optimizer_view, optimizer_gcrnn], [lr_scheduler_view, lr_scheduler_gcrnn]
        