from __future__ import absolute_import
from builtins import map
from builtins import zip
from collections import namedtuple
import pandas as pd
import numpy as np
import pyroaring as pr
from builtins import object
from .sampler_utils import *
# from .evaluation_tools import *

import itertools as itt
import functools
from tqdm.auto import tqdm

def get_combos(b1):
    return pd.DataFrame(b1.param_names)

def get_results(b1, name_method=None):
    params = pd.DataFrame(b1.param_names).drop('object_class', axis=1).apply(lambda x: '_'.join(map(str,x)), axis=1)

    if name_method is not None:
        method = pd.DataFrame(b1.param_names).drop('object_class', axis=1).apply(name_method, axis=1)
    else:
        method = params

    dfs = []
    for (i,(param_names, grader, sampler)) in enumerate(zip(b1.param_names, b1.graders, b1.samplers)):
        pi = params.loc[i]
        mi = method.loc[i]

        rh = grader.get_result_history()
        idcs = rh.reset_index().groupby('ni').index.agg([min, max]).stack().sort_values().values
        assert idcs[0] == 0
        assert idcs[-1] == rh.index[-1]
        rh = rh.iloc[idcs].reset_index(drop=True)
        rh = rh.assign(**param_names)
        rh['bench_index'] = i
        rh['method'] = mi
        rh['params'] = pi
        dfs.append(rh)
    return pd.concat(dfs, axis=0, sort=True, ignore_index=True)


def gen_skip_ilocs(self, skip_time):
    random_offset_ts = (self.gt.index.round('1 ms')
                        + pd.to_timedelta(np.random.randint(9999999), unit='ms'))
    # common orders
    sktim = skip_time.replace(' ', '')
    skip_ilocs = self.gt.assign(sec=random_offset_ts.floor(skip_time)).reset_index().drop_duplicates(
        subset=['sec'], keep='first').index.rename('skip_' + sktim)
    return skip_ilocs