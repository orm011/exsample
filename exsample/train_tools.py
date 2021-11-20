import pytorch_lightning as pl
from pytorch_lightning import Callback
import ray.tune as tune
import numpy as np
import sklearn as sk
import torch

from sklearn.model_selection import StratifiedKFold

class ImageCV(StratifiedKFold):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def split(self, X, y, groups=None):
        y = y.max(axis=(1, 2))
        if len(y.shape) > 1 and y.shape[-1] > 1:
            y = y[..., -1]
        return super().split(X, y)

from torch.utils.data import TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import Subset,ConcatDataset,DataLoader
from sklearn.utils import compute_class_weight

class Pytorch2SKLearn(object):
    def __init__(self, estimator, config):
        self.estimator = estimator  # constructor
        self.config = config  # hparams

        self.estimator_ = None  # None
        self.tr_ = None

    @classmethod
    def _sk2pytorch(self, X):
        return torch.from_numpy(X).permute(0, 3, 1, 2).float()

    @classmethod
    def balanced_pos_weight(self, yval, pooled):
        if pooled:
            yval = yval.flatten(start_dim=2).max(dim=2).values.view(-1)
        else:
            yval = yval.view(-1)

        assert len(yval.shape) == 1
        w0, w1 = compute_class_weight('balanced', classes=[0, 1], y=yval.numpy())
        pos_weight = w1 / w0
        return pos_weight

    def fit(self, X, y, **kwargs):
        config = self.config.copy()

        ys = self._sk2pytorch(y)
        if ys.sum() == 0:
            return self
            ## no positive example, ignore.

        Xs = self._sk2pytorch(X)
        train_ds = TensorDataset(Xs, ys)

        if self.config['pos_weight'] == 'balanced':
            pos_weight = self.balanced_pos_weight(ys, pooled=config['pooled_supervision']),
            config['pos_weight'] = pos_weight

        self.estimator_ = self.estimator(config)
        mod = self.estimator_

        if 'Xval' in kwargs:
            Xval = self._sk2pytorch(kwargs['Xval'])
            yval = self._sk2pytorch(kwargs['yval'])
            val_ds = TensorDataset(Xval, yval)
            num_sanity_val_steps = yval.shape[0] / config['val_batch_size']
            limit_val_batches = num_sanity_val_steps
            extra_cbs = [InitialMetrics()]
            weights_summary = 'top'
            progress_bar_refresh_rate = 1
        else:
            val_ds = train_ds
            num_sanity_val_steps = 0
            limit_val_batches = 0
            extra_cbs = []
            weights_summary = None
            progress_bar_refresh_rate = 0

        cbs = [pl.callbacks.LearningRateLogger()] #if config['optimizer'] != 'LBFGS'
        self.tr_ = pl.Trainer(logger=pl.loggers.TensorBoardLogger(config['log_dir']),
                              callbacks=cbs + extra_cbs,
                              gpus=config['gpus'],
                              max_epochs=config['max_epochs'],
                              num_sanity_val_steps=num_sanity_val_steps,
                              limit_val_batches=limit_val_batches,
                              row_log_interval=1,
                              log_save_interval=1,
                              progress_bar_refresh_rate=progress_bar_refresh_rate,
                              weights_summary=weights_summary)

        train_dl = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True, num_workers=config['num_workers'])

        val_dl = DataLoader(val_ds, batch_size=config['val_batch_size'],
                             num_workers=config['num_workers'])

        self.tr_.fit(self.estimator_, train_dl, val_dl)
        ## flush to tensorboard...
        mod.logger.save()
        mod.logger.experiment.flush()

        return self

    def predict_proba(self, X, pool=True):
        Xs = self._sk2pytorch(X)
        y = self.estimator_.to('cpu').eval().score(Xs).numpy().squeeze(-1)
        return np.stack([1 - y, y], axis=-1)


class TuneReportCallback(Callback):
    def __init__(self, metrics=['val_loss', 'frameAP', 'top_k', 'ndcg_score']):
        self.metrics = metrics

    def on_sanity_check_end(self, trainer, pl_module):
        self.on_validation_end(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        rep = {k: v.item() for (k, v) in trainer.callback_metrics.items() if k in self.metrics}
        tune.report(**rep)


class InitialMetrics(Callback):
    def on_sanity_check_end(self, trainer, pl_module):
        trainer.logger.log_metrics(trainer.callback_metrics, step=-1)


def int_choice(lst):
    return tune.sample_from(lambda _ : int(np.random.choice(lst)))


class MultilabelKFold(StratifiedKFold):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def split(self, X, y, groups=None):
        y = y[..., -1:]  # max(axis=(1, 2))
        return super().split(X, y)


class ImageCV(StratifiedKFold):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def split(self, X, y, groups=None):
        y = y.max(axis=(1, 2))
        if (len(y.shape) > 1) and y.shape[-1] > 1:
            y = y[..., -1:]
        return super().split(X, y)