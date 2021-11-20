from __future__ import division
from __future__ import absolute_import
from collections import deque, OrderedDict, Counter
import string
from .sampler_common import *


def simple_flip(s, bits_needed=63):
    fmt = '0{bn}b'.format(bn=bits_needed)
    s = format(s, fmt)
    sl = list(s)
    sl.reverse()
    return string.atoi(string.joinfields(sl, ''), base=2)


def bytewise_flipbits(b):
    assert b.dtype == np.dtype('uint8')
    return (np.bitwise_and(b * 0x0202020202, 0x010884422010) % 1023).astype('uint8')

def msb2lsb(arr):
    assert arr.dtype == np.uint64, 'use unsigned types'
    tmp0 = arr.copy()
    tmp1 = tmp0.byteswap(inplace=True)
    tmp2 = bytewise_flipbits(tmp1.view('uint8'))
    ret = tmp2.view(np.uint64)

    a0 = np.binary_repr(arr[0], width=64)
    r0 = np.binary_repr(ret[0], width=64)
    assert r0 == ''.join(reversed(a0)), "test case failed"

    return ret

def time_series_to_lsb(frame_ts):
    fts = frame_ts
    tss = fts.asi8.astype('uint64')
    f1 = msb2lsb(tss)
    return pd.Series(f1, index=frame_ts)

def time_split(df, duration='30 min'):
    total_sec = pd.to_timedelta(duration).total_seconds()
    random_offset = pd.to_timedelta(np.random.randint(total_sec), unit='s')
    idx2 = df.index + random_offset
    grid = idx2.round(duration) - random_offset
    return pd.Series(pd.Categorical(grid).rename('time_split_{}'.format(duration.replace(' ', ''))))

def create_split(column, num_splits=190, quantile=True):
    if quantile:
        boundaries = column.quantile(q=np.arange(num_splits + 1) / 1.0 / num_splits)
        cutoffs = boundaries.unique()
    else:
        m = column.min()
        M = column.max()
        step = (M - m) / 1. / num_splits
        cutoffs = np.arange(start=column.min(), stop=column.max(), step=step)

    cutoffs = np.concatenate([[np.NaN], cutoffs, [np.inf]])
    by_speed = column.sort_values(na_position='first')

    cutoff_pos = 1
    # curr_cutoff = cutoffs[cutoff_pos]
    next_cutoff = cutoffs[cutoff_pos]
    partitions = np.zeros_like(by_speed, dtype=np.float)

    for i, val in enumerate(by_speed.values):
        if np.isnan(val):  # Nan case
            partitions[i] = np.NaN
            continue

        while val > next_cutoff:
            cutoff_pos += 1
            # curr_cutoff = cutoffs[cutoff_pos]
            next_cutoff = cutoffs[cutoff_pos]

        assert val <= next_cutoff
        partitions[i] = next_cutoff

    speed_split = pd.Series(partitions, index=by_speed.index)
    speed_split = speed_split.sort_index().rename(
        '%s_split_%d_%s' % (column.name, num_splits, 'Q' if quantile else 'V'))
    return speed_split


std_samples = [100, 300, 600, 1000, 3000, 6000, 10000, 30000]


def bandit_dists_asof(sampler, nsamples=std_samples, splits=None, normalize=True):
    choices = np.array(sampler.choice_history)
    rows = []
    for N in nsamples:
        rows.append(Counter(choices[:N]))

    ans = pd.DataFrame(rows, index=nsamples).fillna(0)
    if splits is not None:
        ans = ans.rename(lambda x: splits[x], axis=1)

    ans = ans.transpose().rename_axis('N samples', axis='columns').rename_axis('choice')
    if normalize:
        ans = ans / 1. / ans.sum()
    return ans.transpose()