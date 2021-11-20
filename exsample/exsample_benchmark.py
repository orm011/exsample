from collections import namedtuple,deque,OrderedDict,defaultdict
import pyroaring as pr
import time
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

def time2frames(duration_str='30 min', frame_rate=30.):
    total_sec = pd.to_timedelta(duration_str).total_seconds()
    return int(np.round(total_sec * frame_rate))

def remove_in(seq, bad_set):
    return seq[~seq.isin(bad_set)]

from .multimap import ArrayMultiMap

from .exsample_dataset import DatasetMetadata,read_data

def standardize_boxes(dm, boxes):  ## use column mapping
    bx = boxes.assign(track_id=pd.Categorical(boxes[dm.track_id]).codes,
                        frame_id=boxes[dm.frame_id],
                        category=boxes[dm.category])[['track_id', 'frame_id', 'category']]

    return bx

def make_default_chunks(boxes, chunk_column, frame_id_column):
    fid = frame_id_column
    chk = chunk_column
    boxcol = boxes[[chunk_column, fid]].groupby(fid).head(n=1).set_index(fid)[chk]
    boxcol = boxcol.sort_index()
    chunk_ids = pd.Categorical(boxcol).codes
    return chunk_ids

class BenchmarkDataset(object):
    def __init__(self, metadata : DatasetMetadata):
        self.metadata = metadata
        self.name = metadata.name
        self._boxes = read_data(self.metadata.boxpath)
        self.boxes = standardize_boxes(self.metadata, self._boxes)
        if metadata.categories is None:
            self.categories = list(self.boxes.category.unique())
        else:
            self.categories = metadata.categories

        if metadata.video_len is None:
            self.video_len = self.boxes.frame_id.max() + 1
        else:
            self.video_len = metadata.video_len

        self.frame_idx = np.arange(self.video_len)

        if self.metadata.default_chunks is None:
            self.default_chunks = self.make_split('30 minutes').rename('default_chunks')
            #self.default_chunks = pd.Series(np.zeros_like()).rename('default_chunks')
        elif isinstance(self.metadata.default_chunks, tuple):
            (file, col) = self.metadata.default_chunks
            chunk_ids = pd.Categorical(read_data(file)[col]).codes
            self.default_chunks = pd.Series(chunk_ids).rename('default_chunks')
        elif isinstance(self.metadata.default_chunks, str):
            chunk_ids = make_default_chunks(self._boxes, metadata.default_chunks, metadata.frame_id)
            self.default_chunks = pd.Series(chunk_ids).rename('default_chunks')
        else:
            assert False

        if self.metadata.logitpath is None:
            self.split = pd.Series(['test' for _ in range(self.video_len)], name='split')
        else:
            self.split = read_data(self.metadata.logitpath)[self.metadata.logitsplit].rename('split')

        self.excluded_set = pr.FrozenBitMap(self.split[self.split.isin(['train', 'val'])].index)
        self.track_maps = {}

        def track_totals(boxes):
            return boxes.groupby(['category', 'track_id']).agg(
                track_length=pd.NamedAgg('frame_id', 'count')).reset_index()

        totals = track_totals(self.boxes)
        valid_tracks = pr.BitMap(totals.track_id[totals.track_length >= 3].values)

        self.ok_boxes = self.boxes[(~self.boxes.frame_id.isin(self.excluded_set))
                                   & (self.boxes.category.isin(set(self.categories)))
                                   & (self.boxes.track_id.isin(valid_tracks))
                                   ]

        self.ok_tracks = track_totals(self.ok_boxes)
        self.NIs = self.ok_tracks.groupby('category').track_id.count()
        self.NF = len(pr.BitMap(range(self.video_len)) - self.excluded_set)  # how much of the dataset
        self.nosplit = pd.Series(np.zeros_like(self.frame_idx)).rename('nosplit')

        for (k, gp) in self.ok_boxes.groupby('category'):
            # for now use dense track ids so we can use array indexing on track id
            dense_track_ids = pd.Categorical(gp.track_id.values).codes
            self.track_maps[k] = ArrayMultiMap(keys=gp.frame_id.values,
                                               values=dense_track_ids)

    def make_split(self, ts):
        if ts == 'nosplit':
            return self.nosplit
        elif ts == 'default':
            return self.default_chunks
        elif isinstance(ts, str):
            #assert (self.default_chunks == 0).all()  # for datasets with substructure, need extra
            return pd.Series(self.frame_idx // time2frames(ts), name=ts)
        else:
            assert False

    def make_distinct_object_grader(self, category):
        return Grader(self.track_maps[category], excluded_set=self.excluded_set)

    def make_object_count_grader(self, category, count):
        return EventGrader(self.ok_boxes, category, count, gap=300)

    def make_order(self, ostr):
        if ostr == 'random':
            arr = np.random.permutation(self.video_len)
        elif ostr == 'random+':
            arr = random_plus(size=self.video_len, scores=None, initial_step=256, initial_offset=None)
        else:
            assert False

        return remove_in(pd.Series(arr, name=ostr), self.excluded_set)


def random_plus(size=None, scores=None, initial_step=256, initial_offset=None):
    if size is None:
        size = scores.shape[0]

    if scores is None:
        scores = np.random.normal(size=size)
    if isinstance(scores, pd.Series):
        scores = scores.values

    x = np.arange(size)
    assert scores.shape[0] == size

    if initial_step > size // 4:
        initial_step = size // 4

    if initial_offset is None:
        initial_offset = np.random.randint(initial_step)

    acc_idxs = []
    step = initial_step
    offset = initial_offset
    seen = pr.BitMap()
    while step >= 1:
        offset = offset % step

        assert offset < step
        tmp = x[offset::step]
        new_set = pr.BitMap(tmp).difference(seen)  # .to_array() # remove duplicates
        next_idxs = np.array(new_set)
        score_order = np.argsort(scores[next_idxs])[-1::-1]  # descending order
        next_idxs = next_idxs[score_order]
        acc_idxs.append(next_idxs)
        seen = seen.union(new_set)

        step = step // 2

    out = np.concatenate(acc_idxs)
    assert np.unique(out).shape[0] == out.shape[0]
    assert out.shape[0] == x.shape[0]
    return out

def getn(v):
    if type(v) in [int, float, str]:
        return v
    elif callable(v):
        if hasattr(v, 'func'):  # assume functools.partial or something like that.
            return v.func.__name__
        elif hasattr(v, '__name__'):
            return v.__name__
    elif v is None:
        return str(None)
    else:
        n = v.name
        assert n is not None
        return n

import itertools as itt
def named_prod(**kwargs):
    ls = list(kwargs.items())
    ks = [k for (k, _) in ls]
    vs = [v for (_, v) in ls]

    for (k, vals) in zip(ks, vs):
        # print(k)
        for v in vals:
            nm = getn(v)  ## make sure there is a name
            # print '  ', nm

    Comb = namedtuple('Comb', ks)
    tuple_it = itt.product(*vs)

    return [Comb(**dict(list(zip(ks, tup))))
            for tup in tuple_it]
def getn(v):
    if type(v) in [int, float, str]:
        return str(v)
    elif callable(v):
        if hasattr(v, 'func'):  # assume functools.partial or something like that.
            return v.func.__name__
        elif hasattr(v, '__name__'):
            return v.__name__
    elif v is None:
        return str(None)
    elif isinstance(v, tuple) or isinstance(v, list):
        return '_'.join([getn(n) for n in v])
    elif isinstance(v, dict):
        return '_'.join([getn(k)+'_' + getn(v) for (k,v) in v.items()])
    elif hasattr(v, 'name'):
        return v.name
    else:
        return str(v)

def named_prod(**kwargs):
    ls = list(kwargs.items())
    ks = [k for (k, _) in ls]
    vs = [v for (_, v) in ls]
    Comb = namedtuple('Comb', ks)
    tuple_it = itt.product(*vs)

    return [Comb(**dict(list(zip(ks, tup))))
            for tup in tuple_it]

from .exsample_sampler import CompositeSampler,PreOrder
import random

class Experiment(object):
    def __init__(self, sampler, grader, params, param_names, NI, NF, batch_size):
        self.sampler = sampler
        self.grader = grader
        self.params = params
        self.param_names = param_names
        self.NI = int(NI)
        self.NF = int(NF)
        self.experiment_id = random.randint(0, 1 << 63)
        self.batch_size= batch_size

    def run_up_to(self, threshold, metric='results', disable_tqdm=False):
        grader = self.grader
        sampler = self.sampler
        assert metric in ['results', 'frames']
        instance_threshold = self.NI
        frames_threshold = self.NF

        if metric == 'results':
            if isinstance(threshold, float):
                assert 0. <= threshold <= 1.
                instance_threshold = min(self.NI, int(np.ceil(self.NI * threshold)))
            elif isinstance(threshold, int):
                instance_threshold = min(self.NI, threshold)
            else:
                assert False
        elif metric == 'frames':
            assert isinstance(threshold, int)
            frames_threshold = min(self.NF, threshold)

        prev_ni = grader.ni()
        recall = grader.ni() / self.NI

        with tqdm(unit='batch', leave=False, disable=disable_tqdm) as it:
            it.set_postfix([('rec', recall), ('ni', grader.ni()),
                            ('nf', grader.nf())], refresh=True)

            while True:
                (ilocs, ids) = sampler.next_batch(n=self.batch_size)
                if ilocs.shape[0] == 0:
                    break

                match_ids = grader.grade_batch(ilocs)
                sampler.feedback((ilocs,ids), match_ids)

                if grader.ni() > prev_ni:
                    prev_ni = grader.ni()
                    recall = grader.ni() / self.NI

                    it.set_postfix([('rec', recall), ('ni', grader.ni()),
                                    ('nf', grader.nf())], refresh=True)

                if (grader.ni() >= instance_threshold or
                        grader.nf() >= frames_threshold):
                    break

    def get_results(self):
        pi = '_'.join(map(str, [v for (k, v) in self.param_names.items() if k not in ['dataset',
                                                                                      'object_class',
                                                                                      'object_count']]))
        mi = pi
        grader = self.grader
        rh = grader.get_result_history()
        idcs = rh.reset_index().groupby('ni').index.agg([min, max]).stack().sort_values().values
        assert idcs[0] == 0
        assert idcs[-1] == rh.index[-1]
        rh = rh.iloc[idcs].reset_index(drop=True)
        rh = rh.assign(**self.param_names)
        rh['experiment_id'] = self.experiment_id
        rh['method'] = mi
        rh['params'] = pi
        rh['instance_f'] = rh.ni / self.NI
        rh['frame_f'] = rh.nf / self.NF
        rh['NI'] = self.NI
        rh['NF'] = self.NF
        return rh

from .exsample_sampler import *

def make_experiment(tup):
    ds = tup.dataset
    grader = BenchmarkDataset.make_distinct_object_grader(ds, tup.object_class)
    params = tup._asdict()

    NI = ds.NIs[tup.object_class]  # bound on instances
    if tup.score_method is None:
        if tup.order == 'random':
            sampler = RandomlyPermutedRange(0, ds.frame_idx.shape[0], excluded_set=ds.excluded_set)
        elif tup.order == 'random+':
            order = BenchmarkDataset.make_order(ds, tup.order).values
            sampler = PreOrder(order, excluded_set=ds.excluded_set)
        else:
            assert False
    elif isinstance(tup.score_method, tuple):
        sm, score_params = tup.score_method
        if isinstance(tup.split, str):
            split = BenchmarkDataset.make_split(ds, tup.split)
        elif isinstance(tup.split, pd.Series):
            split = tup.split
        else:
            assert False

        order = BenchmarkDataset.make_order(ds, tup.order)
        assert order.shape[0] <= split.shape[0]
        samplers = []
        df = pd.DataFrame({'split_id': pd.Categorical(split[order]).codes,  # want dense ids
                           'frame_id': order,  # should be global, ok not dense
                           'order': np.arange(order.shape[0])})

        df = df.sort_values(['split_id', 'order'])
        for (sid, gp) in df.groupby('split_id'):
            ## pd.groupby preserves order within groups, and orders keys
            frame_idxs = gp.frame_id.values
            samplers.append(PreOrder(ilocs=frame_idxs, excluded_set=ds.excluded_set))

        sampler = CompositeSampler(samplers=samplers,
                                   NI=ds.NIs[tup.object_class],
                                   score_method=sm,
                                   score_opts=score_params)

        params.update(score_params)

    param_names = OrderedDict([(k, getn(v)) for (k, v) in params.items()])
    return Experiment(sampler, grader, tup, param_names, NI=NI, NF=ds.NF, batch_size=tup.batch_size)

def make_composite(df, cols):
    df = df[cols]
    tot = df[df.columns[0]]
    for c in df.columns[1:]:
        tot = tot + '_' + df[c].map(str)
    return tot


def savings_at(records, task_keys, method_keys,
               fixed_metric,
               fixed_metric_level,
               measured_metric,
               quantiles=[.1, .5, .9],
               experiment_key='experiment_id', reference_method='None_random_nosplit'):
    assert fixed_metric in ['nf', 'ni']
    assert measured_metric in ['nf', 'ni']

    if fixed_metric_level < 1:
        records

    pred = records[fixed_metric] >= fixed_metric_level
    records = records.assign(metric=records[measured_metric])

    sc = records[pred].groupby(task_keys + method_keys + ['experiment_id'],
                               as_index=False).metric.min()

    def quantile_agg(q):
        return lambda x: np.quantile(x, q)

    msc = sc.groupby(task_keys + method_keys).agg(lower=pd.NamedAgg('metric', quantile_agg(quantiles[0])),
                                                  metric=pd.NamedAgg('metric', quantile_agg(quantiles[1])),
                                                  upper=pd.NamedAgg('metric', quantile_agg(quantiles[2]))
                                                  )
    msc = msc.reset_index()
    gps = []
    for (ks, gp) in msc.groupby(task_keys):
        row = (gp[method_keys] == reference_method).all(axis=1)
        assert row.sum() <= 1, 'must have atmost one reference row per group'
        if row.sum() > 0:
            ref = gp[row].metric.iloc[0]
        else:
            ref = np.nan

        other = gp.metric
        gp = gp.assign(ratio=(ref + 1) / (other + 1))
        gp = gp.assign(ref=ref)
        gps.append(gp)

    return pd.concat(gps, ignore_index=True)


import torch.distributions as dist
import torch

class SimulatedMatchIds(object):
    def __init__(self, Ps, frames_per_chunk):
        self.Ps = Ps
        self.frames_per_chunk = frames_per_chunk

    def __getitem__(self, item):
        j = int(item) // int(self.frames_per_chunk)
        x = dist.Bernoulli(self.Ps[:,j]).sample()
        return torch.where(x)[0]

class Grader(object):
    def __init__(self, frame2track, excluded_set):
        self.match_ids = frame2track
        self.excluded_set = excluded_set
        self.frames_seen = pr.BitMap()
        self.instances_seen = pr.BitMap()

        Hist = namedtuple('Hist', 'timestamp ni nf')
        self.stats = Hist(timestamp=deque([]),
                          ni=deque([]),  # instances so far
                          nf=deque([]),  # frames so far
                          )
        self.record_stats()  ## first row to stats.

    def nf(self):
        return len(self.frames_seen)

    def ni(self):
        return len(self.instances_seen)

    def record_stats(self):
        self.stats.nf.append(self.nf())
        self.stats.ni.append(self.ni())
        self.stats.timestamp.append(time.time())

    def grade_batch(self, new_frames):
        matches = [self.match_ids[i] for i in new_frames]  # also returned to user

        instances = pr.BitMap.union(*[pr.BitMap(m) for m in matches])
        new_instances = pr.BitMap(instances) - self.instances_seen

        if len(new_instances) > 0:
            self.record_stats()

        frame_set = pr.BitMap(new_frames)
        assert len(frame_set) == len(new_frames), 'repeated frame within batch'
        assert frame_set.intersection_cardinality(self.excluded_set) == 0, 'frame id is excluded'
        assert frame_set.intersection_cardinality(self.frames_seen) == 0, 'repeated frame'

        self.frames_seen.update(frame_set)
        self.instances_seen.update(new_instances)

        if len(new_instances) > 0:
            self.record_stats()

        return matches

    def get_result_history(self):
        self.record_stats()  ## add one at this point
        rh = pd.DataFrame(self.stats._asdict())
        rh['wall_clock'] = rh.timestamp - rh.iloc[0].timestamp
        return rh


class EventGrader(object):
    def __init__(self, boxes, object_class, geq_quantity, gap,
                 debug=False):
        self.boxes = boxes
        counts = boxes[boxes.category == object_class].groupby('frame_id').size()
        quals = counts[counts >= geq_quantity]
        self.quals = quals
        self.is_pos = pr.FrozenBitMap(self.quals.index)
        self.GAP = gap

        self.frames_excluded = pr.BitMap() # all frames near previous hits, excludes repeated
        self.frames_seen = pr.BitMap() # all seen frames, including with non-results
        self.instances_seen = pr.BitMap() # frames with hits
        self.excluded_samples = 0
        self.debug = debug

        Hist = namedtuple('Hist', 'timestamp ni nf')
        self.stats = Hist(timestamp=deque([]),
                          ni=deque([]),  # instances so far
                          nf=deque([]),  # frames so far
                          )
        self.record_stats()  ## first row to stats.

        self._check_rep()

    def nf(self):
        return len(self.frames_seen)

    def ni(self):
        return len(self.instances_seen)

    def record_stats(self):
        self.stats.nf.append(self.nf())
        self.stats.ni.append(self.ni())
        self.stats.timestamp.append(time.time())

    @staticmethod
    def excluded_range(idx, gap):
        return max(0,idx-gap), idx + gap + 1

    def _check_rep(self, really):
        if really:
            assert (np.diff(np.array(self.instances_seen)) > self.GAP).all()
            assert self.instances_seen.difference_cardinality(self.frames_excluded) == 0
            assert len(self.instances_seen)*(self.GAP*2+1) >= len(self.frames_excluded)

    def grade(self, idx):
        assert idx not in self.frames_seen
        self.frames_seen.add(idx)

        if idx in self.frames_excluded:
            self.excluded_samples += 1
            if self.excluded_samples > 0.5*self.nf():
                print('warning: many samples are for excluded regions')

            query = pr.BitMap()
            query.add_range(*self.excluded_range(idx, self.GAP))
            repeated = self.instances_seen.intersection(query)
            assert len(repeated) > 0, 'should have neighbors if excluded'
            return np.array(repeated) # should return idx of already sampled result?

        if idx in self.quals:
            self.instances_seen.add(idx)
            self.frames_excluded.add_range(*self.excluded_range(idx, self.GAP))
            self.record_stats()
            self._check_rep(self.debug)
            return np.array([idx]) # use frame to identify event instance

        return np.array([])

    def grade_batch(self, idxs):
        match_ids = []
        for idx in idxs:
            match_ids.append(self.grade(idx))

        return match_ids

    def get_result_history(self):
        self.record_stats()  ## add one at this point
        rh = pd.DataFrame(self.stats._asdict())
        rh['wall_clock'] = rh.timestamp - rh.iloc[0].timestamp
        return rh