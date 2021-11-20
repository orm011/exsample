import torch
import h5py
import pandas as pd

import torchvision.transforms as transforms
import numpy as np
import PIL
import json
import os
from collections import namedtuple

from .dataset_tools import HCatDataset,DataFrameDataset,ZarDataset

def ndict(**kwargs):
    return namedtuple('NT', kwargs.keys())(**kwargs)

def make_bdd_ts(scale=1.):
    target_size = int(720 * scale)
    ts = transforms.Compose([
        # transforms.ToPILImage(), # assume reading image already...
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return ts


class BDDCounts(torch.utils.data.Dataset):
    def __init__(self, split, label_path='/nvme_drive/orm/bdd_label_counts.parquet',
                 xforms=lambda x: x):
        self.label_path = label_path
        self.split = split
        self.xforms = xforms
        all_labs = pd.read_parquet(label_path)

        is_train = split == 'train'
        labs = all_labs.loc[is_train]
        self.labs = labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        return (self.xforms(self.labs.iloc[idx]),)


class BDDActivationsZar(torch.utils.data.Dataset):
    def __init__(self, train_or_val,
                 root='/nvme_drive/orm/bdd10k_activations/',
                 xforms=lambda x: x):
        self.train_or_val = train_or_val
        self.xforms = xforms
        path = '{}/{}.zarr'.format(root,train_or_val)
        self.hdf = ZarDataset(path)

    def __len__(self):
        return len(self.hdf)

    def __getitem__(self, idx):
        data = self.hdf[idx]
        return (self.xforms(torch.from_numpy(data)),)

class BDDImages(torch.utils.data.Dataset):
    def __init__(self, subset, root='/big_fast_drive/orm/bdd_dataset/bdd100k/',
                 xforms=lambda x: x, wrap_output=False):
        self.root = root
        self.subset = subset
        self.imroot = root + '/images/100k/' + subset + '/'
        self.wrap_output = wrap_output
        assert subset in ['train', 'val']


        fpq = pd.read_parquet('/big_fast_drive/orm/bdd_dataset/bdd100k/labels/bdd_image_order.parquet')
        self.files = fpq[fpq.split == self.subset].file.values
        # self.files = sorted(os.listdir(self.imroot)).
        # doesn't match labels for some training set examples. use explicit order from labels.
        self.xforms = xforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = PIL.Image.open(self.imroot + fname)

        if self.wrap_output:
            return (self.xforms(im),)
        else:
            return self.xforms(im)

from torch.utils.data import ConcatDataset,Subset
class BDDAllImages(torch.utils.data.Dataset):
    def __init__(self, root='/big_fast_drive/orm/bdd_dataset/bdd100k/'):
        self.train = BDDImages('train',root=root)
        self.val = BDDImages('val', root=root)
        dummy = Subset(self.train, np.arange(137))  # repeat bc of missing images in train annotations
        self._merged = ConcatDataset([self.train, dummy, self.val])

    def __getitem__(self, idx):
        if idx >= 70000 - 137:
            print('Warning: accessing errored elt')
        return self._merged[idx]

#bdd_root = '/big_fast_drive/orm/bdd_dataset/bdd100k/'
#bdd_root = '/home/gridsan/omoll/data/bdd/bdd100k'
import copy

class BDDLabels(torch.utils.data.Dataset):
    def __init__(self, subset, root='/big_fast_drive/orm/bdd_dataset/bdd100k/', xforms=lambda x: x):
        self.root = root
        self.subset = subset
        self.imroot = root + '/images/100k/{}/'.format(subset)
        self.labroot = root + '/labels/bdd100k_labels_images_{}.json'.format(subset)
        self.xforms = xforms
        self.labels = json.load(open(self.labroot))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = copy.deepcopy(self.labels[idx]) # in case tx mutates data
        return (self.xforms(label),)

class BDDActivationsCoarse(torch.utils.data.Dataset):
    def __init__(self, normalize=True, xforms=None,
                 path='/nvme_drive/orm/car_stuff/vroom_nbs/first10k5by9.npy',
                 stat_path='/nvme_drive/orm/car_stuff/vroom_nbs/first10k5by9_stats.npz'):
        self.activations = np.load(path)
        n = Normalizer2(stat_path) if normalize else (lambda x: x)
        xforms = (lambda x: x) if xforms is None else xforms
        self.xforms = transforms.Compose([n, xforms])
        self.d = 144

    def __len__(self):
        return self.activations.shape[0]

    def __getitem__(self, idx):
        return (self.xforms(self.activations[idx]),)

class BDDActivations8by14(torch.utils.data.Dataset):
    def __init__(self, split, normalize=True, xforms=None,
                 stat_path='/nvme_drive/orm/car_stuff/vroom_nbs/train10k8by15_stats.npz'):
        paths = {'train': '/nvme_drive/orm/car_stuff/vroom_nbs/train10k8by14.npy',
                 'val': '/nvme_drive/orm/car_stuff/vroom_nbs/val10k8by14.npy'}

        self.activations = np.load(paths[split])
        n = Normalizer2(stat_path) if normalize else (lambda x: x)
        xforms = (lambda x: x) if (xforms is None) else xforms
        self.xforms = transforms.Compose([n, xforms])
        self.d = 3 * 32

    def __len__(self):
        return self.activations.shape[0]

    def __getitem__(self, idx):
        return (self.xforms(self.activations[idx]),)


class BDDBoxesParquet(torch.utils.data.Dataset):
    def __init__(self, split, xforms=None):
        self.labdir = '/big_fast_drive/orm/bdd_dataset/bdd100k//labels/'
        fname = {'train': 'bdd100k_labels_images_train.parquet',
                 'train_10k': 'bdd100k_labels_images_train_10k.parquet',
                 'val': 'bdd100k_labels_images_val.parquet'}

        self.labpath = self.labdir + fname[split]
        self.boxes = pd.read_parquet(self.labpath)
        self.xforms = (lambda x: x) if xforms is None else xforms
        self.length = self.boxes.image_id.max() + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.xforms(self.boxes[self.boxes.image_id == idx].copy()),)

class BddBoxesT(object):
    def __init__(self, object_predicate=lambda _: True):
        self.object_predicate = object_predicate

    def __call__(self, bdd_lab):
        return [ob['box2d'] for ob in bdd_lab['labels'] if self.object_predicate(ob)]



from torch.utils.data import DataLoader
def ds2Xy(ds, num_workers=4):
    elts = []
    for (bidx, elt) in enumerate(DataLoader(ds, batch_size=100, num_workers=num_workers)):
        if bidx == 0:
            for i in range(len(elt)):
                elts.append(deque())

        for i in range(len(elt)):
            elts[i].append(elt[i])

    lout = []
    for l in elts:
        lout.append(torch.cat(tuple(l)).numpy())

    return lout


class Normalizer(object):
    def __init__(self, filename):
        arrs = np.load(filename)
        self.mean = torch.from_numpy(arrs.get('mean')).unsqueeze(-1).unsqueeze(-1)
        self.std = torch.from_numpy(arrs.get('std')).unsqueeze(-1).unsqueeze(-1)

    def __call__(self, vec):
        return (vec - self.mean) / self.std

class Normalizer2(object):
    def __init__(self, filename):
        arrs = np.load(filename)
        self.mean = arrs.get('mean')
        self.std = arrs.get('std')

    def __call__(self, vec):
        return (vec - self.mean) / self.std

from torch.utils.data import TensorDataset, ConcatDataset, Subset
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from shapely.geometry import Polygon, LineString, box
from collections import deque

def row2path(row):
    vertices = row.vertices
    types = row.types
    closed = row.closed

    moves = {'L': Path.LINETO,
             'C': Path.CURVE4}
    points = [v for v in vertices]
    codes = [moves[t] for t in types]
    codes[0] = Path.MOVETO
    if closed:
        points.append(points[0])
        codes.append(Path.CLOSEPOLY)
    path = Path(points, codes)
    return path


def row2shapely(row, flipY=False):
    if np.isnan(row.x1):
        assert np.isnan(row.y1)
        path = row2path(row)
        p = path.to_polygons(closed_only=False)
        assert len(p) == 1, 'unexpected data. may need to create collection'
        pts = p[0]
        if flipY:
            pts[:, 1] = -pts[:, 1]

        if row.closed:
            ans = Polygon(pts)
        else:
            ans = LineString(pts)
    else:
        # assert np.isnan(row.vertices)
        if flipY:
            ans = box(row.x1, -row.y1, row.x2, -row.y2)
        else:
            ans = box(row.x1, row.y1, row.x2, row.y2)

    return ans

def make_geometry(df, flipY=False):
    df = df[['vertices', 'types', 'closed', 'x1', 'y1', 'x2', 'y2']]
    acc = deque([])
    for r in df.itertuples():
        acc.append(row2shapely(r, flipY))
    return pd.Series(acc, index=df.index, name='geom')


def poly2patch(row, alpha=1., scale=1., color='red'):
    path = row2path(row)
    return PathPatch(path,
                     facecolor=color if row.closed else 'none',
                     edgecolor=color,  # if not closed else 'none',
                     lw=1 if row.closed else 2 * scale, alpha=alpha,
                     antialiased=False, snap=True)


import pandas as pd
from .imgviz import *

def show_annotations(boxdf, image=None, mask=None, show_grid=False, d=3*32):
    w, h = image.size

    if 'geometry' not in boxdf.columns:
        boxdf = boxdf.assign(geometry=make_geometry(boxdf, flipY=True))

    dpi = 80  # plt.rcParams['figure.dpi'] # 80 used for BDD in bdd code. different from plt. default.
    gpobj = (ggimg(image, data=boxdf)
             + geom_map(mapping=aes(geometry='geometry', color='category'),
                        data=boxdf[boxdf.annotation_type == 'box'],
                        alpha=.5, size=1, fill=None)
             + geom_map(mapping=aes(geometry='geometry', color='laneType'),
                        data=boxdf[boxdf.annotation_type == 'line'],
                        alpha=.5, size=2)
             + geom_map(mapping=aes(geometry='geometry', color='areaType'),
                        data=boxdf[boxdf.annotation_type == 'polygon'],
                        alpha=.3, size=1, fill=None)
             # + geom_text(mapping=aes(label='category', color='category', x='x1', y='y1'), size=6)
             )

    if show_grid:
        gpobj = (gpobj + annotate('vline', xintercept=np.arange(0, w, step=d), color='white', alpha=.3)
                 + annotate('hline', yintercept=np.arange(0, h, step=d), color='white', alpha=.3))

    ## need to return figure, b/c
    f = gpobj.draw()
    if image is not None:
        f = add_image(f, image)

    if mask is not None:
        f = add_mask(f, mask, d=d)

    return f

def grid_intercepts(xy1, xy2, xarray, yarray):
    '''
    :param xy1: segment start
    :param xy2: segment end
    :param xarray: x grid
    :param yarray: y grid
    :return: positions (x,y) along grid where segment intercepts it.
    Helpful to get grid mask of line annotations.
    '''
    (x1, y1) = xy1
    (x2, y2) = xy2

    alphas = (xarray - x2) / (x1 - x2)
    ys = alphas * y1 + (1 - alphas) * y2
    xcrossings = np.stack([xarray, ys], axis=-1)

    alphas2 = (yarray - y2) / (y1 - y2)
    xs = alphas2 * x1 + (1 - alphas2) * x2
    ycrossings = np.stack([xs, yarray], axis=-1)

    return np.sort(np.concatenate([xcrossings, ycrossings]), axis=0)

def segment2ij(xy1, xy2, d):
    """ convert segment to grid mask"""
    (x1, y1) = xy1
    (x2, y2) = xy2
    xmin = np.floor(min(x1, x2) / d)
    xmax = np.floor(max(x1, x2) / d)
    ymin = np.floor(min(y1, y2) / d)
    ymax = np.floor(max(y1, y2) / d)
    xrng = np.arange(xmin, xmax + 1)
    yrng = np.arange(ymin, ymax + 1)
    assert xrng.shape[0] > 0
    assert yrng.shape[0] > 0

    if xrng.shape[0] > 1 and yrng.shape[0] > 1:  # use slopes
        pts = grid_intercepts(xy1, xy2, xrng * d, yrng * d)
        pts = np.round(pts / d, decimals=2)  # round odd ends before applying floor
        jis = np.abs(np.floor(pts)).astype('int')  # some -0. show up.
        # on the boundaries, some solutions may show up beyond the range of the other dimension
        valid_j = (jis[:, 0] >= xmin) & (jis[:, 0] <= xmax)
        valid_i = (jis[:, 1] >= ymin) & (jis[:, 1] <= ymax)
        jis = jis[valid_i & valid_j]
        ijs = jis[:, ::-1]
    elif xrng.shape[0] == 1:
        js = np.concatenate([xrng] * yrng.shape[0])
        ijs = np.stack([yrng, js], axis=-1)
    elif yrng.shape[0] == 1:
        iis = np.concatenate([yrng] * xrng.shape[0])
        ijs = np.stack([iis, xrng], axis=-1)

    return ijs

def lineString2ij(ln, d):  # go seg by seg (some lines aren't straight)
    """convert linestring (multiple segments) to grid mask"""
    sln = ln.simplify(tolerance=d / 2)  # simplify to make life easier
    xys = np.stack(sln.xy, axis=1)

    acc = deque([])
    for (xy1, xy2) in zip(xys[:-1], xys[1:]):
        ijs = segment2ij(xy1, xy2, d)
        acc.append(ijs)

    return np.concatenate(acc)

class Line2MaskT(object):
    def __init__(self, d, image_size):  # d is fixed if the model is fixed
        self.d = d
        (w, h) = image_size
        self.nh = (h + self.d - 1) // self.d
        self.nw = (w + self.d - 1) // self.d

    def __add_line(self, mask, line):
        ijs = lineString2ij(line, self.d)
        # some ijs may exceed

        mask[ijs[:, 0], ijs[:, 1]] = 1

    def __call__(self, lns):
        mask = torch.zeros(self.nh, self.nw)
        for ln in lns:
            self.__add_line(mask, ln)
        return mask

class Boxes2MaskT(object):
    def __init__(self, d, image_size):  # d is fixed if the model is fixed
        self.d = d
        (w, h) = image_size
        self.nh = (h + self.d - 1) // self.d
        self.nw = (w + self.d - 1) // self.d

    def __add_box(self, mask, box):
        assert tuple(box.keys()) == ('x1', 'y1', 'x2', 'y2')
        xmin, ymin, xmax, ymax = tuple(box.values())
        xlim = int(xmin // self.d), int((xmax + self.d - 1) // self.d)
        ylim = int(ymin // self.d), int((ymax + self.d - 1) // self.d)
        mask[ylim[0]:ylim[1], xlim[0]:xlim[1]] = 1

    def __call__(self, box_list):
        mask = torch.zeros(self.nh, self.nw)
        for box in box_list:
            self.__add_box(mask, box)
        return mask


class MakeMasks(object):
    def __init__(self, d, categories):
        self.d = d
        self.categories = categories
        self.line_categories = ['crosswalk']
        self.b2m = Boxes2MaskT(d=self.d, image_size=(1280, 720))
        self.l2m = Line2MaskT(d=self.d, image_size=(1280, 720))

    def __call__(self, boxes):
        masks = []
        for c in self.categories:
            bx = boxes[(boxes.category == c) | (boxes.laneType == c)]

            if c not in self.line_categories:
                bxlist = bx[['x1', 'y1', 'x2', 'y2']].to_dict(orient='records')
                m = self.b2m(bxlist)
            else:
                geom = make_geometry(bx)
                m = self.l2m(geom)

            masks.append(m)

        return torch.stack(masks, dim=-1)


def make_composite_ys(eb2, combo, split, num_workers=4, d=3*32):
    def xform_mask(trainmasks_):  # helper for person riding.
        person = (trainmasks_[..., 1] > 0) | (trainmasks_[..., 2] > 0)
        person = person.astype('float')
        ytrain_ = np.stack([trainmasks_[..., 0], person, trainmasks_[..., -1]], axis=-1)
        return ytrain_

    if combo == 'person_riding_bike':
        box_xforms = transforms.Compose(
            [MakeMasks(d=d, categories=['bike', 'person', 'rider', 'person_riding_bike'])])
        box_train = DataFrameDataset(eb2[(eb2.split == split)], index_var='image_id', max_idx=9999, xforms=box_xforms)
        trainmasks_ = ds2Xy(box_train, num_workers=num_workers)[0]
        ytrainbike_ = xform_mask(trainmasks_)
        return ytrainbike_
    elif combo == 'person_riding_motor':
        box_xforms = transforms.Compose(
            [MakeMasks(d=d, categories=['motor', 'person', 'rider', 'person_riding_motor'])])
        box_train = DataFrameDataset(eb2[(eb2.split == split)], index_var='image_id', max_idx=9999, xforms=box_xforms)
        trainmasks_ = ds2Xy(box_train, num_workers=num_workers)[0]
        ytrainbike_ = xform_mask(trainmasks_)
        return ytrainbike_
    elif combo == 'person_crossing':
        box_xforms = transforms.Compose(
            [MakeMasks(d=d, categories=['person', 'crosswalk', 'person_crossing'])])
        box_train = DataFrameDataset(eb2[eb2.split == split], index_var='image_id', max_idx=9999, xforms=box_xforms)
        trainmasks_ = ds2Xy(box_train, num_workers=num_workers)[0]
        return trainmasks_
    else:
        assert False, combo