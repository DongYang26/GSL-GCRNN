import traceback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import models
import tasks
import utils.callbacks
import utils.data
import utils.logging
from utils.params import set_params
import pandas as pd
import numpy as np
from pytorch_lightning import seed_everything
import warnings

seed = 42
seed_everything(seed, workers=True)
warnings.filterwarnings('ignore')


def get_model(args, dataTraVal):
    model = None
    if args.model_name == "GSLGCRNN":
        model = models.GSLGCRNN(dataTraVal, arg=args, hidden_dim=args.hidden_dim)
    return model


def get_task(args, model, dm):
    task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
        model=model, feat_max_val=dm.feat_max_val, **vars(args))
    return task


def get_callback():
    checkpoint_callback_2 = pl.callbacks.ModelCheckpoint(monitor="train_loss_2")
    early_stop_2 = pl.callbacks.EarlyStopping(monitor='train_loss_2',min_delta=0.0,patience=64,verbose=False,mode='min')
    callbacks = [checkpoint_callback_2, early_stop_2]
    return callbacks


def main_supervised(args):
    dataTraVal = utils.data.SpatioTemporalCSVDataModule(**vars(args))
    model = get_model(args, dataTraVal)
    task = get_task(args, model, dataTraVal)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=get_callback(), fast_dev_run=False, deterministic=True)
    results = trainer.fit(task, dataTraVal)
    # trainer.validate(dataloaders=dataTraVal)
    return results


def main(args):
    rank_zero_info(vars(args))
    results = globals()["main_" + args.settings](args)
    return results


if __name__ == "__main__":
    args = set_params()
    utils.logging.format_logger(pl._logger)
    if args.log_path is not None:
        utils.logging.output_logger_to_file(pl._logger, args.log_path)
    try:
        results = main(args)
        print(results)
    except:
        traceback.print_exc()
