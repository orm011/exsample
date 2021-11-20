import torch
import torchvision
import json
from torchvision import transforms

from .query import display_images
import json
import glob
import os
import sklearn as sk
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score, make_scorer, average_precision_score
from sklearn.linear_model import LogisticRegression

import joblib
import PIL


# acts = []
# with torch.no_grad():
#     for (i,elt) in enumerate(tqdm(ims)):
#         im = PIL.Image.open('/big_fast_drive/orm/coco/val2017/' + elt['file_name'])
#         tim = coco_ts(im)
#         act = r50(tim.unsqueeze(0).to(device)).squeeze(0).to('cpu').numpy().transpose(1,2,0)
#         acts.append(act)


# flat activations:
# shps = []
# image_ids = []
# flat_acts = []
# for (i, act) in enumerate(acts2):
#     (h, w) = act.shape[0], act.shape[1]
#     sz = h * w
#     shps.append((act.shape[0], act.shape[1]))
#     image_ids.extend([i] * sz)
#     flt = act.reshape(-1, act.shape[-1])
#     flat_acts.append(flt)
#
# acts2 = np.concatenate(flat_acts)
# ids2 = np.array(image_ids)
# shps2 = np.array(shps)


to_rgb = transforms.Lambda(lambda im : im.convert(mode='RGB'))
coco_ts = bdd_ts = transforms.Compose([
        to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class DB(object):
    def __init__(self, filenames, activations, flat_activations, ids, shapes, mod, device, tx):
        self.filenames = filenames
        self.activations = activations  # indexable by id
        self.flat_activations = flat_activations  # flat array
        self.ids = ids
        self.shapes = shapes
        self.tx = tx
        self.mod = mod.to(device)
        self.device = device


class Query(object):
    def __init__(self, db, seed_folder):
        self.db = db
        self.seed_folder = seed_folder
        self.im_paths = []
        self.ims = []
        self.__init_images()
        assert len(self.ims) > 0

        self.augmented_ims = []
        self.__augment_images()
        assert len(self.augmented_ims) > 0

        self.seed_Xss = []
        self.seed_yss = []
        self.__init_Xy()
        assert len(self.seed_Xss) == len(self.augmented_ims)
        assert len(self.seed_yss) == len(self.augmented_ims)

        ### updated during loop
        self.Xss = []
        self.yss = []
        self.indices = []
        self.seen = set([])
        self.models = []

    def __init_images(self):
        imfs1 = glob.glob(self.seed_folder + '/*jpeg')
        imfs2 = glob.glob(self.seed_folder + '/*jpg')
        imfs3 = glob.glob(self.seed_folder + '/*png')
        imfs = sorted(imfs1 + imfs2 + imfs3)
        self.im_paths = imfs
        self.ims = [PIL.Image.open(imf).convert(mode='RGB') for imf in self.im_paths]

    def __augment_images(self):
        aug_tx = transforms.Compose([
            transforms.RandomHorizontalFlip(p=.5),
            transforms.RandomAffine(degrees=20, scale=(.7, 1 / .7))
        ])

        for im in tqdm(self.ims):
            for _ in range(5):  # 5 random samples to enrich im.
                rim = aug_tx(im)
                self.augmented_ims.append(rim)

    def __init_Xy(self):
        with torch.no_grad():
            for rim in self.augmented_ims:
                exim = self.db.tx(rim)
                ex = exim.unsqueeze(0).to(self.db.device)
                xft = self.db.mod(ex).to('cpu').numpy().transpose(0, 2, 3, 1)
                self.seed_Xss.append(xft)
                self.seed_yss.append(np.ones(1))

    def getXy(self, use_seed_data=True):
        Xss = self.seed_Xss + self.Xss if use_seed_data else self.Xss
        yss = self.seed_yss + self.yss if use_seed_data else self.yss

        flatX = np.concatenate([X.mean(axis=(1, 2)) for X in Xss])
        Xs = flatX
        ys = np.concatenate(yss)
        return Xs, ys



def sampling_loop(q1, batch_size=9, ground_truth=None, use_seed_data=True):
    #    search things with high similarity score
    #    take top10 and show them
    #    have them be labelled
    if len(q1.models) == 0:
        next_idxs = list(np.random.permutation(len(q1.db.filenames))[:batch_size])
    else:  # scoring everything...
        mod = q1.models[-1]
        Xs, ys = q1.getXy(use_seed_data)

        if ys.sum() == 0:
            # continue sampling until we find true examples
            idxs = np.random.permutation(len(q1.db.filenames))
        else:
            # otherwise fit model...
            mod.fit(Xs, ys)
            m = q1.models[-1]
            mp = m.predict_proba(q1.db.flat_activations)[:, 1]
            mp2 = pd.DataFrame({'p': mp, 'frame_id': q1.db.ids})
            sk = mp2.groupby('frame_id').mean()['p'].values
            idxs = np.argsort(sk)[::-1]

        next_idxs = []
        for i in idxs:
            if len(next_idxs) == batch_size:
                break

            if i in q1.seen:
                continue
            else:
                next_idxs.append(i)

    q1.indices.extend(next_idxs)
    for i in next_idxs:
        q1.seen.add(i)

    next_pics = [PIL.Image.open(q1.db.filenames[idx]) for idx in next_idxs]
    Xs = [q1.db.activations[idx] for idx in next_idxs]

    if ground_truth is None:  # display and label
        labs = display_images(next_pics)
    else:  # use given labels
        labs = ground_truth[next_idxs]

    q1.Xss.extend(Xs)
    q1.yss.append(labs)
    mod = LogisticRegression(class_weight='balanced', max_iter=10000)
    q1.models.append(mod)


def get_counts(q1):
    ys = np.concatenate(q1.yss)
    pos = np.where(ys)[0]
    npos = pos.shape[0]
    total = ys.shape[0]
    return (npos, total, pos)


def run_loop(db, qstr, limit, repeat=3, **sampling_kw_args):
    res = []
    queries = []

    for _ in tqdm(range(repeat)):
        q = Query(db, qstr)
        queries.append(q)

        while True:
            print('.', end='')
            sampling_loop(q, **sampling_kw_args)
            (npos, total, pos) = get_counts(q)

            if npos >= limit:
                res.append(pos[:limit])
                print('found {:d} after {:d}'.format(npos, pos[limit - 1]))
                break

    return res, queries


# coco_objects = pd.DataFrame.from_records(val_annotation['categories'])
# cocod = { k:v for (k,v) in zip(coco_objects.id,coco_objects.name)}
# cts = coco_val_annot.groupby(['image_id', 'category_id']).size()
# ctotals = cts.reset_index()[['image_id', 'category_id']].groupby('category_id').size()
# val_counts = pd.Series(ctotals.values, index=coco_objects.name)
# coco_val_counts = cts.unstack(level=1).rename(mapper=lambda x : cocod[x], axis=1)
# cts.reset_index().category_id.describe()