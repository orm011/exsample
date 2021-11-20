from builtins import zip
from builtins import range
from builtins import object
import cv2
import hwang
import multiprocessing
import time
from collections import OrderedDict
import torch.utils.data
import torch.utils
import numpy as np
from .avutils import VideoDataset
import geopandas as gpd
import pandas as pd
import shapely
import os
import glob

from shapely import geometry as geom
import shapely.wkt


def to_unicode_columns(df):
    df = df.rename(axis=1, mapper=lambda x: x.decode("utf-8"))
    return df

def read_pq(*args, **kwargs):
    pq = pd.read_parquet(*args, **kwargs)
    if isinstance(pq.columns[0], bytes):
        return to_unicode_columns(pq)

    return pq


cam_holder = geom.box(minx=624, maxx=856, miny=443,maxy=693) # eg. see frame 412029, fire hydrant
yi_logo = geom.box(minx=8, maxx=90, miny=11,maxy=92)


lowbox = shapely.wkt.loads('''POLYGON ((468.5799 243.79, 
        468.5799 400.138,
         29.4594 400.138, 
         29.4594 243.79, 
         468.5799 243.79))''')

per_dataset_tweaks = {
    'jackson-square-long':{
        'truck': dict(label=8), # ok.
        'bicycle':dict(label=2), #pretty ok. 1/9 error with car @lowbox.
        'bus':dict(label=6), # ok. minivans and some pickup trucks.
        #'motorcycle':dict(label=4), # mostly confused with cars waiting at the bottom.., or bikes on cars.

        'person':dict(label=1), # ok. 2/10 error.
        'car':dict(label=3), #ok.
    },
    'dashcam_unified_ng': {
        'stop sign':dict(label=12, ratio_gt=.8, width_gt=20.,
                                      bad_box=yi_logo, bad_box_threshold=.8),
        'parking meter':dict(label=13, ratio_gt=2.),
        'traffic light':dict(label=10, ratio_gt=2., width_gt=20.),
        'truck': dict(label=8, ratio_gt=.2),
        'bicycle':dict(label=2, ratio_lt=1.),
        'bus':dict(label=6, height_gt=40., width_gt=20., ratio_lt=1.1),
        # bus is one of the more error prone categories. eg. buildings get confused w. bus.
        #'motorcycle':dict(label=4), # mostly wrong, don't use.
        #'bench':dict(label=14),# mostly wrong, don't use.
        'fire hydrant':dict(label=11, bad_box=cam_holder, bad_box_threshold=.7), # works pretty well
        'person':dict(label=1, ratio_gt=2., width_gt=25.),
        'car':dict(label=3, ratio_gt=1., width_gt=25.),
    }
}

COCO_CATEGORIES = [
    "__background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow", ]

bdd_categories = ['person',
 'rider',
 'car',
 'bus',
 'truck',
 'bike',
 'motor',
 'traffic light',
 'traffic sign',
 'train']

ds2label = {
    'dashcam_unified_ng':COCO_CATEGORIES,
    'jackson-town-square':COCO_CATEGORIES,
    'bdd1k':bdd_categories
}



## classes actually used in experiments
ds2class = dict(
    [
        ('dashcam_unified_ng',[
            'person',
            'bicycle',
          #      'car', # too common
          #      'motorcycle', # not accurate enough
                'bus',
                'truck',
                'traffic light',
                'fire hydrant',
                'stop sign',
          #      'parking meter'
                ]),
        ('coral-reef-long', ['person']),
        ('jackson-town-square', ['person',
                'bicycle',
                'car',
                'motorcycle',
                'bus',
                'truck',]
         ),
        ('bdd1k',
         ['person',
          'rider',
          #'car', # too common
          'bus',
          'truck',
          'bike',
          'motor',
          'traffic light',
          'traffic sign',])
    ]
)

ds2threshold = dict([
    ('dashcam_unified_ng', .2),
    ('coral-reef-long', .2), # used in that paper
    ('jackson-town-square', .2),
    ('bdd1k', .5)
])

from collections import OrderedDict

# COCO categories for pretty print
# copy pasted from maskrcnn code used to generate boxes
COCO_CATEGORIES = [
 "__background",
 "person",
 "bicycle",
 "car",
 "motorcycle",
 "airplane",
 "bus",
 "train",
 "truck",
 "boat",
 "traffic light",
 "fire hydrant",
 "stop sign",
 "parking meter",
 "bench",
 "bird",
 "cat",
 "dog",
 "horse",
 "sheep",
 "cow",
 "elephant",
 "bear",
 "zebra",
 "giraffe",
 "backpack",
 "umbrella",
 "handbag",
 "tie",
 "suitcase",
 "frisbee",
 "skis",
 "snowboard",
 "sports ball",
 "kite",
 "baseball bat",
 "baseball glove",
 "skateboard",
 "surfboard",
 "tennis racket",
 "bottle",
 "wine glass",
 "cup",
 "fork",
 "knife",
 "spoon",
 "bowl",
 "banana",
 "apple",
 "sandwich",
 "orange",
 "broccoli",
 "carrot",
 "hot dog",
 "pizza",
 "donut",
 "cake",
 "chair",
 "couch",
 "potted plant",
 "bed",
 "dining table",
 "toilet",
 "tv",
 "laptop",
 "mouse",
 "remote",
 "keyboard",
 "cell phone",
 "microwave",
 "oven",
 "toaster",
 "sink",
 "refrigerator",
 "book",
 "clock",
 "vase",
 "scissors",
 "teddy bear",
 "hair drier",
 "toothbrush",
]

COCO_IDS = OrderedDict([(v,k) for (k,v) in enumerate(COCO_CATEGORIES)])

import shapely.geometry as geom
from tqdm import tqdm_notebook as tqdm
import time


def match_detections(ba, frame_id_col='frame_idx', label_col='label',
                     minx='minx', miny='miny', maxx='maxx', maxy='maxy', iou_min=.2):
    import scipy
    import scipy.optimize
    import pygeos
    baboxes = pygeos.box(*[ba[c] for c in [minx, miny, maxx, maxy]])
    orig_index = ba.index

    assert ba.index.is_unique
    ba = ba.assign(box_id=np.arange(ba.shape[0]))
    ba = ba.assign(box=baboxes)
    ba = ba.assign(area=pygeos.area(ba.box))
    ba = ba.assign(label=ba[label_col], frame_idx=ba[frame_id_col])
    ba = ba.reset_index()

    ba = ba[['box_id', 'box', 'area', 'label', 'frame_idx']]  # avoid extra cols...

    active_tracks = np.array([], dtype=np.int)
    tids = ba.box_id.values.copy()  # each box is initially its own track

    ## ds: active objects: each has its own id and its last frame_id seen.
    ## added when a new box does not match anything in it
    ## any object that hasn't been seen in 5 frames is ended
    ## objects never really change tid, they just may be rejected at the end.
    GRACE_PERIOD = 5
    COST_MAX = 0
    n_unique = ba.frame_idx.unique().shape[0]

    for (frame_idx, detections) in tqdm(ba.groupby('frame_idx'),
                                        total=n_unique):
        # clean any stale detections
        active_frames = ba.frame_idx.values[active_tracks]
        still_active = (frame_idx - active_frames <= GRACE_PERIOD)
        active_tracks = active_tracks[still_active]

        active_dets = ba.iloc[active_tracks]
        didx, aidx = np.meshgrid(np.arange(detections.shape[0]), np.arange(active_dets.shape[0]))
        detbox = detections.box.values[didx]
        abox = active_dets.box.values[aidx]

        a_intersect = pygeos.area(pygeos.intersection(detbox, abox))
        a_union = detections.area.values[didx] + active_dets.area.values[aidx] - a_intersect
        iou = a_intersect / a_union
        label_mismatch = detections.label.values[didx] != active_dets.label.values[aidx]

        cost_matrix = COST_MAX - iou
        cost_matrix[label_mismatch] = COST_MAX
        cost_matrix[iou < iou_min] = COST_MAX

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
        cost_ok = cost_matrix[row_ind, col_ind] < COST_MAX
        row_ind = row_ind[cost_ok]
        col_ind = col_ind[cost_ok]

        new_track = (np.ones(detections.shape[0]) > 0)
        new_track[col_ind] = False

        tids[detections.box_id.values[col_ind]] = tids[active_dets.box_id.values[row_ind]]
        active_tracks[row_ind] = detections.box_id.values[col_ind]

        active_tracks = np.concatenate([active_tracks, detections.box_id.values[new_track]])

    return pd.Series(orig_index[tids], index=orig_index)


def add_box_data(idf, box_blacklist=[]):
    df = idf
    df = df.assign(width=df.maxx - df.minx, height=df.maxy - df.miny)
    df = df.assign(ratio=df['height'] / df['width'], area=df['height'] * df['width'])

    boxes = df[['minx', 'maxx', 'miny', 'maxy']].apply(lambda row: geom.box(**row), axis=1)
    boxes = gpd.GeoSeries(boxes)
    ious = []

    for (i, box) in enumerate(box_blacklist):
        a = box.area
        intr = boxes.intersection(box).area
        union = df['area'] + a - intr
        ious.append(intr / union)

    return df, ious


def filter_tracks(b70, label, szs_gt=2, ratio_gt=.1, ratio_lt=np.inf, width_gt=0., height_gt=0.,
                  bad_boxes=[]):
    b70 = b70[(~b70.tid.isna()) & (b70.label == label)]
    assert b70.shape[0] > 0

    #     badbox = b70.apply(lambda r : riou(r, bad_box), axis=1)
    #     box_ok = (badbox < bad_box_threshold)
    #     assert box_ok.sum() > 0

    #     szs = b70.detections.apply(len)
    #     rtio = b70.apply(ratio,axis=1)
    #     wdth = b70.apply(width, axis=1)
    #     hght = b70.apply(height, axis=1)
    ok = []
    for tid, df1 in b70.groupby('tid'):
        df, ious = add_box_data(df1, [b for (b, _) in bad_boxes])
        median_metric = df.median()

        if (df.shape[0] > szs_gt
                and median_metric.height > height_gt
                and median_metric.width > width_gt
                and median_metric.ratio > ratio_gt
                and median_metric.ratio < ratio_lt):

            iou_ok = True
            for (iou, (_, iou_lt)) in zip(ious, bad_boxes):
                iou_ok = iou_ok and (iou.median() < iou_lt)

            if iou_ok:
                ok.append(df)

    return pd.concat(ok)


def init_gt(boxes,video):
    gb = boxes.sort_values(['label', 'frame_idx']).reset_index().set_index(['label', 'frame_idx', 'index'])

    def process_label(gb, label_id):
        tids = gb.loc[label_id].tid
        x = pd.Series(pd.Categorical(tids).codes + 1, index=tids.index)
        arr = x.groupby('frame_idx').apply(np.array)
        return arr

    # process_label
    gts = []
    for i in gb.index.levels[0]:
        g = process_label(gb, i)
        gts.append(g)

    lb = pd.concat(gts, keys=gb.index.levels[0]).unstack('label').reindex(pd.RangeIndex(len(video)))
    return lb.rename(mapper=lambda x: COCO_CATEGORIES[x], axis=1)


class Dataset(object):
    def __init__(self, data_root, dataset_name, gt=None):
        self.DATA_ROOT = data_root
        self.dataset_name = dataset_name
        self.__max_scores = None
        self.__gt = gt
        self.__video = None
        self.max_scores_path = '{r}/csv/{ds}_maxscores.parquet'.format(r=self.DATA_ROOT,
                                                                       ds=self.dataset_name)
        self.threshold = ds2threshold[self.dataset_name]
        self.data_csv_path = '{r}/csv/{ds}.csv'.format(r=self.DATA_ROOT, ds=self.dataset_name)
        self.video_path = '{r}/videos/{ds}_kf20_full.mp4'.format(r=self.DATA_ROOT, ds=self.dataset_name)
        self.video = VideoDataset(self.video_path, transform=tocv2)
        self.video_len = self.video.len
        self.boxpath = '{r}/csv/{ds}.track_ids.clean.parquet'.format(r=self.DATA_ROOT, ds=self.dataset_name)
        boxes = pd.read_parquet(self.boxpath)
        self.boxes = boxes.sort_values(['label','frame_idx', 'tid']).set_index(['label','frame_idx'])
        self.raw_boxes = boxes.reset_index(drop=True)


    def list_methods(self):
        path = '/{r}/sampling_experiments/{ds}/'.format(r=self.DATA_ROOT, ds=self.dataset_name)
        return os.listdir(path)

    def list_results(self, method):
        path = '/{r}/sampling_experiments/{ds}/{method}/'.format(r=self.DATA_ROOT,
                                                                 ds=self.dataset_name,
                                                                 method=method)

        return sorted(os.listdir(path), reverse=True)


    def results_path(self, method, date=None):
        dates = self.list_results(method)
        if date == None:
            date = dates[0]

        assert date in dates, dates
        expdir = '/{r}/sampling_experiments/{ds}/{method}/{date}'.format(r=self.DATA_ROOT,
                                                                         ds=self.dataset_name,
                                                                         method=method,
                                                                         date=date)

        import re
        names = glob.glob(expdir + '/grader*.parquet')
        maxpos = np.argmax([int(re.split('(\d+)', os.path.basename(x))[1]) for x in names])
        return names[maxpos].replace('//', '/')

    def results(self, method, date=None):
        return read_pq(self.results_path(method,date))

    def history_path(self, method, date=None):
        return self.results_path(method, date).replace('grader', 'hist')

    def history(self, method, date=None):
        rawdf = read_pq(self.history_path(method,date))
        if 'ilocs' in rawdf.columns:
            rawdf['frame'] = rawdf['ilocs']
        if 'iloc' in rawdf.columns:
            rawdf['frame'] = rawdf['iloc']

        rawdf['frame'] = rawdf.frame.astype('int')
        return rawdf.set_index('frame')


def load_noscopecsv(nsf):
    with open(nsf, 'r') as f:
        nsparams = f.readline()
        nsparams2 = f.readline()

    (start, end) = [int(x) for x in os.path.basename(nsf).split('.')[0].split('_')[1:]]
    vals = nsparams.strip().replace('#', '').strip().replace(',', '').replace(':', '').split(' ')[:-5]
    vals2 = nsparams2.strip().replace('#', '').strip().replace(',', '').replace(':', '').split(' ')

    params = OrderedDict(list(zip(vals[::2], [float(v) for v in vals[1::2]])) + list(zip(vals2[::2], vals2[1::2])))
    params['start'] = start
    params['end'] = end
    rc = pd.read_csv(nsf, skiprows=2).rename(axis=1, mapper={'# frame': 'frame'})
    rc = rc.assign(frame=rc.frame - 1 + start)
    return rc, pd.Series(params)


class Noscope(object):

    kDistRes = (50, 50)
    kDiffRes = (100, 100)

    def __init__(self, dataset, acc='0.1', object_class='person'):
        self.ds = dataset
        self.acc = acc
        self.object_class = object_class
        self.test_csv_path = self._test_csv_path()
        res, params = load_noscopecsv(self.test_csv_path)
        mbasename = os.path.basename(params.model)
        params['model'] = '{}/cnn-models/{}'.format(dataset.DATA_ROOT, mbasename)
        self.params = params
        self.test_results = res[res.status >= 2].set_index('frame')
        noscope_avg_path = '{}/cnn-avg/{}.txt'.format(self.ds.DATA_ROOT, self.ds.dataset_name)
        avg_frm = np.loadtxt(noscope_avg_path)
        self.avg_frm = avg_frm.reshape(list(self.kDistRes) + [3])
        self._history = None


    def _test_csv_path(self):
        pattern = '{rt}/experiments/{ds}/{acc}/test*csv'.format(rt=self.ds.DATA_ROOT,
                                                                ds=self.ds.dataset_name,
                                                                acc=self.acc)
        candidates = glob.glob(pattern)
        if len(candidates) != 1:
            raise Exception('no matching noscope csv for {pat}'.format(pat=pattern))

        noscope_output_csv = candidates[0]
        return noscope_output_csv

    @property
    def history(self): #not available before we have actually run an experiment
        if self._history is None:
            self._history = self.ds.history('noscope')
        return self._history

    # def ns_accuracy(self):
    #     return ns.main(yolo_csv_filename=self.ds.data_csv_path,
    #                    TEST_START_IDX=self.params.start,
    #                    TEST_END_IDX=self.params.end,
    #                    noscope_csv_filename=self.test_csv_path,
    #                    object_name=self.object_class
    #                    )

def tocv2(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def smooth_scores(nms2, window):
    """ assumes 30 fps
    """
    assert window % 2 == 1, "odd window works better"
    assert nms2.index.is_monotonic_increasing
    assert nms2.index.is_unique
    nms2 = nms2.fillna(0)

    if window <= 2:
        return nms2

    assert window//2 >= 1
    nms2 = nms2.rolling(window=window, min_periods=window//2,
                                       win_type='gaussian', center=True).mean(std=window//2)
    return nms2


def label_runs(col, run_limit=None):
    if type(run_limit) is str:
        run_limit = pd.to_timedelta(run_limit)
        assert col.index.dtype

    idx = col.index
    if idx.dtype.kind == 'i':
        idx = idx.astype('float')

    labels = np.zeros_like(idx)
    run_start = None
    for (i, sc) in enumerate(col.values):
        if sc:
            if run_start is None:  # start new run
                run_start = idx[i]
            elif run_limit is not None and idx[i] - run_start > run_limit:  # break run
                run_start = idx[i]

        if not sc:  # run interrupted
            run_start = None

        labels[i] = run_start

    out = pd.Series(labels, index=col.index, name=col.name)
    assert (~out.isna() == col).all()
    return out



def score2instance(scores, nonsmooth, score_threshold, run_limit=None):
    result_cols = []
    for c in scores.columns:
        oc = label_runs(scores[c] > score_threshold,
                        run_limit=run_limit)

        # remove entries w low score
        oc[nonsmooth[c] <= score_threshold] = None
        result_cols.append(pd.Categorical(oc))

    data = OrderedDict([(c, oc) for (c, oc) in zip(scores.columns, result_cols)])
    return pd.DataFrame(data, index=scores.index)


def instance_gt_heuristic(maxscores,
                          smoothing_window=None,
                          score_threshold=None,
                          classes=None,
                          max_instance_duration=None):

    assert smoothing_window is not None
    assert score_threshold is not None
    assert score_threshold <= 1. and score_threshold >= 0.

    nonsmooth = maxscores.fillna(0)
    smscores = smooth_scores(nonsmooth.reset_index(drop=True), window=smoothing_window).set_index(maxscores.index)
    match_ids = score2instance(smscores, nonsmooth, score_threshold=score_threshold, run_limit=max_instance_duration)
    return match_ids