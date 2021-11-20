import torch
import torchvision
import PIL
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms as t
from torchvision import transforms
import pandas as pd
import numpy as np
import glob

import ipywidgets as widgets
import functools


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet, pooling=None):
        super().__init__()

        self.resnet = resnet
        del self.resnet.fc
        del self.resnet.avgpool
        self.pooling = pooling if (pooling is not None) else (lambda x: x)
        self.eval()  # set to eval.

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.pooling(x)
        return x

    @classmethod
    def get_xform(self):
        return transforms.Compose([
            transforms.Lambda(lambda x: x.convert(mode='RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def extract(self, im):
        with torch.no_grad():
            ac = self.get_xform()(im)
            ret = self(ac.unsqueeze(dim=0)).squeeze(dim=0)
            return ret


bdd_ts = transforms.Compose([
        transforms.Lambda(lambda x : x.convert(mode='RGB')),
        transforms.Resize(500),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def random_rescale(im):
    factor = np.power(2., np.random.uniform(-2, 2))
    shorter_side = min(im.size)
    new_size = int(shorter_side * factor)
    new_size = int(np.clip([new_size], a_min=30, a_max=400))
    return t.Resize(new_size)(im)

from io import BytesIO
import ipywidgets as widgets
import functools
from .bdd_tools import *

class DB(object):
    def __init__(self, device):
        enc = nn.Sequential(
            ResNetFeatureExtractor(torchvision.models.resnet50(pretrained=True)),
            nn.AvgPool2d(kernel_size=(3, 3), ceil_mode=True)
        )
        self.d = 144  # est input pixels => output activation.

        self.imtx = transforms.Compose([
            transforms.Lambda(lambda x: x.convert(mode='RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.enc = enc.to(device).eval()
        self.activations = np.load('first10k5by9.npy')
        #         stpath = '/big_fast_drive/bdd_activations/bdd_resnet18_mean_std.npz'
        self.act_tx = transforms.Compose([lambda x: x.permute(1, 2, 0).numpy()])

        self.basedir = '/big_fast_drive/orm/bdd_dataset/bdd100k/images/100k/train/'
        self.images = BDDImages('train')
        self.filenames = self.images.files[:len(self.activations)]
        self.dl = torch.utils.data.DataLoader(self.activations, batch_size=50,
                                              shuffle=False, num_workers=10)
        self.device = device
        self.b2m = Boxes2MaskT(self.d, image_size=(1280, 720))

    def __len__(self):
        return len(self.activations)

    def encode(self, im):
        with torch.no_grad():
            X = self.imtx(im).to(self.device).unsqueeze(dim=0)
            y = self.enc(X).to('cpu').squeeze(dim=0)
            yp = self.act_tx(y)

        return yp

    def run_scorer(self, scorer):
        sk = scorer.score_batch(self.activations)
        return sk


class DB2(object):
    def __init__(self, device):
        enc = ResNetFeatureExtractor(torchvision.models.resnet18(pretrained=True))
        self.imtx = transforms.Compose([
            transforms.Lambda(lambda x: x.convert(mode='RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.d = 32  ## est. pixels at center of single output activation

        self.enc = enc.to(device).eval()
        stpath = '/big_fast_drive/bdd_activations/bdd_resnet18_mean_std.npz'
        self.act_tx = transforms.Compose([
            Normalizer(stpath),
            lambda x: x.permute(1, 2, 0).numpy()])

        self.activations = BDDActivationsZar('train', xforms=self.act_tx)
        self.basedir = '/big_fast_drive/orm/bdd_dataset/bdd100k/images/100k/train/'
        ims = BDDImages('train')
        self.images = ims
        self.filenames = ims.files[:len(self.activations)]

        self.device = device
        self.dl = torch.utils.data.DataLoader(self.activations, batch_size=50,
                                              shuffle=False, num_workers=10)
        self.b2m = Boxes2MaskT(self.d, image_size=(1280, 720))

    def __len__(self):
        return len(self.activations)

    def encode(self, im):
        with torch.no_grad():
            X = self.imtx(im).to(self.device).unsqueeze(dim=0)
            y = self.enc(X).to('cpu').squeeze(dim=0)
            yp = self.act_tx(y)

        return yp

    def run_scorer(self, scorer):
        scores = []
        for (batch,) in self.dl:
            sk = scorer.score_batch(batch)
            scores.append(sk)
        return np.concatenate(scores)



def display_images(im_list, COLS=3, label=True):
    ROWS = int(np.ceil(len(im_list) / COLS))
    rows = []
    labels = np.zeros(len(im_list), dtype=np.int)

    def on_click(b, index):
        if labels[index] == 0:
            labels[index] = 1
            b.description = '✅'
        else:
            labels[index] = 0
            b.description = '❌'

    for row in range(ROWS):
        cols = []
        for col in range(COLS):
            index = row * COLS + col
            if index >= len(im_list):
                continue

            b = BytesIO()
            im0 = im_list[index]
            im0 = transforms.Resize(150)(im0)
            imw, imh = im0.size

            im0.save(b, format='png')

            image = widgets.Image(value=b.getvalue())

            if label:
                button = widgets.Button(layout=widgets.Layout(width='{}px'.format(imw),
                                                              height='20px'))
                # Bind the click event to the on_click function, with our index as argument
                button.on_click(functools.partial(on_click, index=index))
                button.description = '❌'

                # Create a vertical layout box, image above the button
                box = widgets.VBox([image, button])
            else:
                box = image

            cols.append(box)

        # Create a horizontal layout box, grouping all the columns together
        rows.append(widgets.HBox(cols))
    # Create a vertical layout box, grouping all the rows together
    result = widgets.VBox(rows)
    IPython.core.display.display(result)
    return labels

import IPython
from tqdm.notebook import tqdm
import sklearn as sk
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression


class Scorer(object):
    def score_batch(self, db):
        raise NotImplementedError

    def retrain(self, qdata):
        raise NotImplementedError

class SkLrScorerFineGrained(Scorer):
    def __init__(self):
        self.fitted = 0
        self.model = LogisticRegression(class_weight='balanced', max_iter=10000)

    def _getXy(self, qdata):
        self = qdata
        seed_x = qdata.seed_Xss
        seed_masks = [np.ones(x.shape[:-1]) for x in qdata.seed_Xss]

        boxes = np.concatenate(qdata.boxes)
        rXss = np.concatenate(qdata.Xss)
        r_masks = []
        r_x = []
        for x, bs in zip(rXss, boxes):
            r_x.append(x)

            bxlist = []
            for b in bs:
                if b[2] == 0:
                    continue
                bxlist.append({'x1': b[0], 'y1': b[1], 'x2': b[0] + b[2], 'y2': b[1] + b[3]})

            r_masks.append(self.db.b2m(bxlist))

        x = seed_x + r_x
        masks = seed_masks + r_masks

        depth = seed_x[0].shape[-1]
        x2 = [t.reshape(-1, depth) for t in x]
        masks2 = [t.reshape(-1) for t in masks]
        Xs = np.concatenate(x2)
        ys = np.concatenate(masks2)
        return Xs, ys

    def score_batch(self, batch):
        if self.fitted == 0:
            return np.random.randn(batch.shape[0])

        batch = batch
        orig_shape = batch.shape
        acts = batch.reshape(-1, batch.shape[-1])
        mp = self.model.predict_proba(acts)[:, 1]
        mp = mp.reshape(*orig_shape[:-1])  # skip channel dim
        sk = mp.mean(axis=(1, 2))
        return sk

    def retrain(self, qdata):
        Xs, ys = self._getXy(qdata)
        model = LogisticRegression(class_weight='balanced', max_iter=10000)
        model.fit(Xs, ys)
        self.model = model
        self.fitted += 1


class SkLrScorerCoarse(Scorer):
    def __init__(self):
        self.fitted = 0
        self.model = LogisticRegression(class_weight='balanced', max_iter=10000)

    def _getXy(self, qdata):
        qXss = sum(qdata.Xss, [])  # concat lists dim = 0.
        Xss = qdata.seed_Xss + qXss
        yss = qdata.seed_yss + qdata.yss

        spatial_avg = [Xs.mean(axis=(0, 1)) for Xs in Xss]
        Xs = np.stack(spatial_avg)
        ys = np.concatenate(yss)

        assert len(Xs.shape) == 2
        assert Xs.shape[0] == ys.shape[0]
        assert Xs.shape[1] == qdata.Xss[0][0].shape[-1]
        return Xs, ys

    def score_batch(self, batch):
        if self.fitted == 0:
            return np.random.randn(batch.shape[0])

        orig_shape = batch.shape
        acts = batch.reshape(-1, batch.shape[-1])
        mp = self.model.predict_proba(acts)[:, 1]
        mp = mp.reshape(*orig_shape[:-1])
        sk = mp.mean(axis=(1, 2))
        return sk

    def retrain(self, qdata):
        Xs, ys = self._getXy(qdata)
        model = LogisticRegression(class_weight='balanced', max_iter=10000)
        model.fit(Xs, ys)
        self.model = model
        self.fitted += 1


class DatasetV0(object):
    def __init__(self, q0):
        self.seed_Xss = q0.seed_Xss
        self.seed_yss = q0.seed_yss

        self.Xss = q0.Xss
        self.yss = q0.yss

        self.Xlist = []
        self.Xlist.extend(q0.seed_Xss)
        for bx in self.Xss:
            arrs = np.split(bx, np.arange(1, bx.shape[0]))
            self.Xlist.extend(arrs)

        self.ylist = []
        self.ylist.extend(q0.seed_yss)
        for by in self.yss:
            arrs = np.split(by, np.arange(1, by.shape[0]))
            self.ylist.extend(arrs)

        assert len(self.Xlist) == len(self.ylist)

    def __len__(self):
        return len(self.Xlist)

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.Xlist[idx].squeeze(axis=0)).float()
        X = tensor.permute(2, 0, 1)
        return X, torch.from_numpy(self.ylist[idx]).float().squeeze(axis=0)

import pytorch_lightning as pl

from vsms.query_model_variants import BinaryFeedback

class PTScorer(Scorer):
    def __init__(self):
        self.config = {'in_features': 2048, 'num_workers': 1, 'object_class': 'person',
                       'train_batch_size': 1, 'val_batch_size': 1, 'lr': .02, 'gamma': .5,
                       'milestones': [3, 6], 'pos_weight': 7}
        self.model = BinaryFeedback(self.config)  # random init

    def score(self, db):
        X = np.transpose(db.activations, axes=(0, 3, 1, 2))
        model = self.model.eval().to('cpu')

        with torch.no_grad():
            ptX = torch.from_numpy(X).float().to(model.device)
            y = model(ptX).sigmoid()
            py = y.to('cpu').numpy()

        return py

    def retrain(self, qdata):
        ds = DatasetV0(qdata)
        dl_train = torch.utils.data.DataLoader(ds, shuffle=True)
        dl_val = torch.utils.data.DataLoader(ds, shuffle=False)

        root_dir = '/nvme_drive/orm/pl_comp/'
        tb_logger = pl.loggers.TensorBoardLogger(root_dir, name='query_demo')
        trainer = pl.Trainer(default_root_dir=root_dir,
                             logger=tb_logger,
                             max_epochs=10,
                             callbacks=[pl.callbacks.LearningRateLogger()],
                             gpus=1,
                             log_gpu_memory=False,
                             row_log_interval=10,
                             early_stop_callback=False,
                             # resume_from_checkpoint=None,# if epoch == 0 else checkpath,
                             weights_summary=None,
                             # fast_dev_run=True,
                             )

        net = BinaryFeedback(self.config)
        trainer.fit(net, dl_train, dl_val)
        self.model = net


from jupyter_innotater import *


class Query(object):
    def __init__(self, db, seed_folder, score_alg):
        self.db = db
        self.scorer = score_alg

        self.seed_folder = seed_folder
        self.im_paths = []
        imfs1 = glob.glob(self.seed_folder + '*jpeg')
        imfs2 = glob.glob(self.seed_folder + '*jpg')
        imfs3 = glob.glob(self.seed_folder + '*png')
        imfs = sorted(imfs1 + imfs2 + imfs3)
        self.im_paths.extend(imfs)

        self.seed_Xss = []
        self.seed_yss = []
        max_boxes = 5
        self.seed_boxes = [np.zeros((len(self.im_paths), max_boxes, 4), dtype='int')]

        self.Xss = []
        self.yss = []
        self.boxes = []

        self.indices = []
        self.seen = set([])
        self.seed_ims = []
        self.xformed_ims = []

        self.augment_tx = transforms.Compose([
            random_rescale,
            transforms.RandomHorizontalFlip()])

        self.__init_labels()

        self.models = []
        self.model = None

    def get_dataset(self):
        pass

    def __init_labels(self):
        self.ims = [PIL.Image.open(imf) for imf in self.im_paths]
        pos_fts = []
        for im in tqdm(self.ims):
            for _ in range(5):  # 5 random samples to enrich im.
                rim = self.augment_tx(im)
                self.xformed_ims.append(rim)
                xft = self.db.encode(rim)
                pos_fts.append(xft)

        self.seed_Xss.extend(pos_fts)
        for ft in pos_fts:
            self.seed_yss.append(np.ones(1))

    def show_seeds(self):
        ims = [PIL.Image.open(imf) for imf in self.im_paths]
        display_images(ims, label=False)

    def sampling_loop(self, n_samples=9, ground_truth=None, use_seed_data=True):
        q1 = self
        if len(self.Xss) > 0:  # assume have positives and negatives.
            self.scorer.retrain(self)
            sk = q1.db.run_scorer(self.scorer)
            idxs = np.argsort(-sk)
        else:  # first
            idxs = np.random.permutation(len(q1.db))

        next_idxs = []
        for i in idxs:
            if len(next_idxs) == n_samples:
                break
            if i in q1.seen:
                continue
            else:
                next_idxs.append(i)

        q1.indices.append(next_idxs)
        for i in next_idxs:
            q1.seen.add(i)

        ## todo:in datasets don't return a tuple... instead return a tensor even for scalars.
        ## otherwise, need to remember everywhere. its not intuitive.
        ## would like to switch from array to dataset without having to change too much.

        next_pics = [q1.db.images[idx][0] for idx in next_idxs]
        Xs = [q1.db.activations[idx] for idx in next_idxs]

        if ground_truth is None:  # display and label
            labs = display_images(next_pics)
        else:  # use given labels
            labs = ground_truth[next_idxs]

        q1.Xss.append(Xs)
        q1.yss.append(labs)
        max_boxes = 5
        targets_bboxes = np.zeros((len(Xs), max_boxes, 4), dtype='int')
        q1.boxes.append(targets_bboxes)

    def show_positives(q1):
        ys = np.concatenate(q1.yss)
        indices = sum(q1.indices, [])
        fidx = [indices[i] for i in np.where(ys)[0]]
        ims = [q1.db.images[fi][0] for fi in fidx]
        new_labs = display_images(ims, label=False)

    def label_seeds(q1):
        fnms = [fi for fi in q1.im_paths]
        boxes = q1.seed_boxes[-1]

        widget = Innotater(
            [ImageInnotation(fnms),  # Display the image itself
             # TextInnotation(fnms, multiline=False) # Display the image filename
             ],
            [
                RepeatInnotation(
                    (BoundingBoxInnotation, boxes),  # Individual animal bounding box
                    max_repeats=boxes.shape[1], min_repeats=1,
                )
            ])
        return widget

    def label_batch_positives(q1):
        ys = q1.yss[-1]
        indices = q1.indices[-1]
        fidx = indices
        fnms = [q1.db.filenames[fi] for fi in fidx]
        boxes = q1.boxes[-1]
        w = 1280 // 2
        h = 720 // 2

        widget = Innotater(
            [ImageInnotation(fnms, path=q1.db.basedir, width=w, height=h),  # Display the image itself
             # TextInnotation(fnms, multiline=False) # Display the image filename
             ],
            [
                RepeatInnotation(
                    (BoundingBoxInnotation, boxes),  # Individual animal bounding box
                    max_repeats=boxes.shape[1], min_repeats=1,
                )
            ])
        return widget