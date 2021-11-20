import torch
import pytorch_lightning as pl
import torch.nn as nn
from sklearn.metrics import average_precision_score
import numpy as np
from .bdd_tools import *

import torchvision.transforms as transforms

from collections import namedtuple

def top_k_prec(y_true, y_pred, k):
    top_k_idx = np.argsort(-y_pred)[:k]
    return y_true[top_k_idx].sum() / k

import pytorch_lightning
from pytorch_lightning.metrics import AveragePrecision
import torchcontrib


class BinaryFeedback(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        hparams2 = ndict(**hparams)
        self.hparams2 = hparams2
        self.num_workers = hparams2.num_workers

        def make_mod():
            m = nn.Sequential(
                nn.Conv2d(in_channels=hparams2.in_features, out_channels=1, kernel_size=1),
            )
            return m

        def get_pool(poolstr):
            if poolstr == 'avg':
                return nn.AdaptiveAvgPool2d(1)
            elif poolstr == 'max':
                return nn.AdaptiveMaxPool2d(1)
            else:
                assert False

        self.mod = make_mod()
        self.train_pool = get_pool(hparams['train_pool'])
        self.eval_pool = get_pool(hparams['eval_pool'])
        self.target_pool = nn.AdaptiveMaxPool2d(1)

        pos_weights = torch.tensor(hparams2.pos_weight).float()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weights)

    def loss(self, out, target):
        if self.hparams['pooled_supervision']:
            target = self.target_pool(target)
            out = self.train_pool(out)

        total_loss = self.loss_fn(out, target)

        return total_loss

    def forward(self, X):
        act = self.mod(X)
        return act

    @torch.no_grad()
    def score(self, X=None, act=None):
        if act is None:
            act = self(X)
        return self.eval_pool(act.sigmoid()).flatten(start_dim=1)

    def configure_optimizers(self):

        if self.hparams['optimizer'] == 'SGD':
            opt = torch.optim.SGD(params=self.parameters(), lr=self.hparams2.lr,
                                  weight_decay=self.hparams2.weight_decay)
            if ('swa_start' in self.hparams and
                    self.hparams['swa_start'] >= 0):
                opt = torchcontrib.optim.SWA(opt, swa_start=self.hparams['swa_start'],
                                             swa_freq=self.hparams['swa_freq'],
                                             )
            self.opt = opt

        elif self.hparams['optimizer'] == 'LBFGS':
            opt = torch.optim.LBFGS(params=self.parameters(), lr=self.hparams2.lr, history_size=10,
                                    line_search_fn='strong_wolfe')

        else:
            assert False

        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.hparams['milestones'],
                                                     gamma=self.hparams['gamma'], last_epoch=-1)

        return [opt], [sched]

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        data, target = batch
        assert len(target.shape) == 4

        logits = self(data)

        total_loss = self.loss(logits, target)
        return {'loss': total_loss, 'log': {'train_loss': total_loss}}

    def validation_step(self, batch, batch_idx):
        x, target = batch
        assert len(target.shape) == 4

        logits = self(x)
        val_loss = self.loss(logits, target)
        yhat = self.score(act=logits)
        ytrue = self.target_pool(target).flatten(start_dim=1)

        ret = {'val_loss': val_loss, 'y_hat': yhat, 'y_true': ytrue}
        return ret

    def validation_epoch_end(self, outputs):
        val_loss = [x['val_loss'] for x in outputs]  # single value per batch
        y_hat = [x['y_hat'] for x in outputs]  # multiple values per batch
        y_true = [x['y_true'] for x in outputs]

        val_loss_mean = torch.stack(val_loss).mean()
        y_pred = torch.cat(y_hat)
        y_true = torch.cat(y_true)
        AP = AveragePrecision()(y_pred.view(-1), y_true.view(-1))  # pl ap has different order
        log = {'val_loss': val_loss_mean,
               'frameAP': AP
               }

        return {
            'val_loss': val_loss_mean,
            'log': log
        }

from .bdd_tools import  BDDActivationsZar

class CoarseDetector(pl.LightningModule):
    def __init__(self, hparams, ds_train, ds_val, num_workers=1):
        super(CoarseDetector, self).__init__()
        self.hparams = hparams
        hparams2 = ndict(**hparams)
        self.hparams2 = hparams2
        self.num_workers = num_workers
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels=hparams2.in_features, out_channels=1,
                      kernel_size=hparams2.ksize, padding=hparams2.ksize // 2),
        )

        self.ds_train = ds_train
        self.ds_val = ds_val
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(hparams2.pos_weight))
        self.best_score = None

    @staticmethod
    def make_ds(train_or_val, object_class):
        label_xform = transforms.Compose([BddBoxesT(lambda ob: ob['category'] == object_class),
                                          Boxes2MaskT(d=32, image_size=(1280, 720))])
        act_xform = Normalizer('/home/gridsan/omoll/data/bdd_activations/bdd_resnet18_mean_std.npz')

        labels = BDDLabels(train_or_val, xforms=label_xform)
        activations = BDDActivationsZar(train_or_val, xforms=act_xform)
        hds = HCatDataset(activations, labels)
        return hds


    def loss(self, out, target):
        return self.loss_fn(out, target)

    def forward(self, X):
        return self.mod(X).squeeze(1)

    def predict_proba(self, X):
        return self.mod(X).sigmoid()

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        data, target = batch
        out = self.forward(data)
        loss = self.loss(out, target)
        #         pnorm = list(self.mod[0].parameters())[0].norm()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_train, batch_size=self.hparams2.batch_size,
                                           num_workers=self.num_workers, shuffle=True)

    def configure_optimizers(self):
        # opt = torch.optim.SGD(params=self.parameters(), lr=self.hparams2.initial_lr)
        opt = torch.optim.SGD(params=self.parameters(), lr=self.hparams2.initial_lr)
        # sched = torch.optim.lr_scheduler.ExponentialLR()
        lr_lambda = lambda x: [1, 1, 1, .33, .33, .33][x] if x < 4 else .1
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda, last_epoch=-1)
        # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.33,patience=2)
        return [opt], [sched]

    def validation_step(self, batch, batch_idx):
        x, ytrue = batch
        ypred = self.forward(x)
        bsize = ypred.shape[0]
        val_loss = self.loss(ypred, ytrue)
        ypred_agg = ypred.sigmoid().view((bsize, -1)).max(dim=-1).values
        ytrue_agg = ytrue.view((bsize, -1)).max(dim=-1).values

        ret = {'val_loss': val_loss, 'y_hat': ypred_agg, 'y_true': ytrue_agg}
        return ret

    def validation_epoch_end(self, outputs):
        val_loss = [x['val_loss'] for x in outputs]  # single value per batch
        y_hat = [x['y_hat'] for x in outputs]  # multiple values per batch
        y_true = [x['y_true'] for x in outputs]

        val_loss_mean = torch.stack(val_loss).mean()

        y_pred = torch.cat(y_hat).view(-1).to('cpu').numpy()
        y_true = torch.cat(y_true).view(-1).to('cpu').numpy()

        mAP = average_precision_score(y_true, y_pred)
        randomAP = average_precision_score(y_true, np.random.randn(y_true.shape[0]))
        dct = {'val_loss': val_loss_mean, 'mAP': torch.tensor(mAP).float(),
               'randomAP': torch.tensor(randomAP).float()}

        log = dct.copy()
        dct['log'] = log
        return dct

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_val, batch_size=50, num_workers=self.num_workers)

import torchvision

class BoxEncoding(object):
    def __init__(self, hparams2):
        # D, box_pw, box_ph, image_size):
        (w, h) = hparams2.image_size
        self.D = hparams2.D
        self.box_pw = hparams2.box_pw
        self.box_ph = hparams2.box_ph
        D = self.D
        ah, aw = (h + D - 1) // D, (w + D - 1) // D
        self.shp = (5, ah, aw)

        iis, jjs = torch.meshgrid(torch.arange(ah), torch.arange(aw))
        self.iis = iis.unsqueeze(0).float()
        self.jjs = jjs.unsqueeze(0).float()

    def _encode_box(self, boxdict):
        D = self.D
        (x1, y1, x2, y2) = list(boxdict.values())
        midx, midy = (x1 + x2) / 2, (y1 + y2) / 2
        boxx, boxy = int(midx // D), int(midy // D)

        rel_x = (midx % D) / D
        rel_y = (midy % D) / D
        rel_w = np.log((x2 - x1) / self.box_pw)
        rel_h = np.log((y2 - y1) / self.box_ph)
        return (boxy, boxx), torch.tensor([1., rel_x, rel_y, rel_w, rel_h])

    def __call__(self, lab):
        res = torch.zeros(self.shp)
        for box in lab:
            (i, j), vec = self._encode_box(box)
            res[:, i, j] = vec
        return res

    def decode(self, sxywh):
        ntout = sxywh
        D = self.D
        assert ntout.shape[1] == 5, 'dim0 is batch. dim1 channel. include scores'

        (scores, relmidx, relmidy, tw, th) = ntout.unbind(1)
        ah, aw = scores.shape[-2:]

        width = tw.clamp(max=8).exp() * self.box_pw
        height = th.clamp(max=8).exp() * self.box_ph
        midx = (self.jjs.to(relmidx.device) + relmidx) * D
        midy = (self.iis.to(relmidy.device) + relmidy) * D

        x1 = midx - width / 2
        x2 = midx + width / 2
        y1 = midy - height / 2
        y2 = midy + height / 2

        xyboxes = torch.stack([x1, y1, x2, y2], dim=1)
        scores = scores.flatten(-2, -1).unsqueeze(-1)
        xyboxes = xyboxes.flatten(-2, -1).transpose(-2, -1)
        return (scores, xyboxes)

    def post_process_batch(self, ypred, nms_threshold,
                           score_threshold, assume_logits):
        assert len(ypred.shape) == 4
        assert ypred.shape[1] == 5  #
        (sc_out, box_out) = self.decode(ypred.to('cpu'))

        if assume_logits:
            sc_out = sc_out.sigmoid().squeeze(-1)
        else:
            sc_out = sc_out.squeeze(-1)

        res = []
        for (scores, boxes) in zip(sc_out, box_out):
            # breakpoint()
            pass_score = (scores > score_threshold)
            boxes = boxes[pass_score]
            scores = scores[pass_score]

            idxs = torchvision.ops.nms(boxes, scores, iou_threshold=nms_threshold)
            scores = scores[idxs]
            boxes = boxes[idxs]
            assert scores.shape[0] == boxes.shape[0]
            res.append({'scores': scores, 'boxes': boxes})
        return res

    @staticmethod
    def box_overlay(boxes=None, pred=None, im=None, composite=True, box_color=(150, 30, 30)):
        assert not (boxes is None and pred is None)
        if boxes is None:
            boxes = [{'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2} for
                     (x1, y1, x2, y2) in list(zip(*pred['boxes'].t().numpy()))]

        TINT_COLOR = (100, 100, 100)
        HIGHLIGHT_COLOR = (150, 150, 150)
        BOX_COLOR = (150, 30, 30)
        TRANSPARENCY = 1.  # Degree of transparency, 0-100%
        OPACITY = int(255 * TRANSPARENCY)

        overlay = PIL.Image.new('RGBA', im.size, TINT_COLOR + (0,))
        draw = PIL.ImageDraw.Draw(overlay)  # Create a context for drawing things on it.

        for b in boxes:
            draw.rectangle((b['x1'], b['y1'], b['x2'], b['y2']),
                           outline=BOX_COLOR + (OPACITY,))

        if im is None:
            return overlay.convert('RGB')
        else:
            return PIL.Image.alpha_composite(im.convert('RGBA'), overlay).convert("RGB")


class DetLoss(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.pos_weight = hparams.pos_weight
        self.weight_loss_box = hparams.loss_box
        self.weight_loss_center = hparams.loss_center
        self.D = hparams.D
        self.L = hparams.clamp_limit
        self.bceloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight), reduction='sum')
        self.mseloss = nn.MSELoss(reduction='sum')
        # NB: mean won't work with empty set. don't want to weight them equally either.

    def __call__(self, out, target):
        out = out.transpose(1, 0).flatten(1).transpose(1, 0)
        target = target.transpose(1, 0).flatten(1).transpose(1, 0)
        factor = 1 / target.shape[0]
        loss_p = self.bceloss(out[:, 0], target[:, 0]) * factor

        active_mask = (target[:, 0] > 0)
        target_centers = target[:, 1:3][active_mask]
        target_sizes = target[:, 3:][active_mask]
        out_centers = out[:, 1:3][active_mask]
        out_sizes = out[:, 3:][active_mask]

        if self.hparams.exp_size:
            loss_sizes = self.mseloss(out_sizes.clamp(max=self.L).exp() * self.hparams.box_pw,
                                      target_sizes.exp() * self.hparams.box_pw) * factor
            loss_center = self.mseloss(out_centers * self.D, target_centers * self.D) * factor
        else:
            loss_sizes = self.mseloss(out_sizes, target_sizes) * factor
            loss_center = self.mseloss(out_centers, target_centers) * factor

        loss_box = loss_sizes + (self.weight_loss_center * loss_center)
        loss = loss_p + (self.weight_loss_box * loss_box)
        # self.weight_loss_size*loss_sizes + self.weight_loss_center*loss_center

        ans = {'loss_prob': loss_p, 'loss_center': loss_center, 'loss_sizes': loss_sizes,
               'loss_box': loss_box, 'loss': loss}
        return ans


class BoxMetrics(object):
    def __init__(self, nms_threshold=.5,
                 proposal_threshold=.5,
                 score_threshold=.01):
        self.score_threshold = score_threshold
        self.proposal_threshold = proposal_threshold
        self.nms_threshold = nms_threshold

    def __call__(self, sctrue, xyxytrue, scpred, xyxypred):
        totals = {'matched_proposals': [],
                  'total_proposals': [],
                  'matched_true': [],
                  'total_true': [],
                  'max_score': []}

        for (st, xyt, sp, xyp) in zip(sctrue, xyxytrue, scpred, xyxypred):
            xytf = xyt[st.squeeze() > 0]
            scores = sp.sigmoid().view(-1)
            pass_score = (scores > self.score_threshold)
            fxyp = xyp[pass_score]
            fscores = scores[pass_score]
            idxs = torchvision.ops.nms(fxyp, fscores, iou_threshold=self.nms_threshold)
            fxyp2 = fxyp[idxs]
            fscores2 = fscores[idxs]
            totals['max_score'].append(scores.max().float().item())
            totals['total_proposals'].append(float(fxyp2.shape[0]))
            totals['total_true'].append(float(xytf.shape[0]))

            if xytf.shape[0] == 0 or fxyp2.shape[0] == 0:
                totals['matched_proposals'].append(0.)
                totals['matched_true'].append(0.)
                continue

            ious = torchvision.ops.box_iou(fxyp2, xytf)
            best_iou_proposed = ious.max(dim=1).values.view(-1)
            best_iou_actual = ious.max(dim=0).values.view(-1)

            mp = (best_iou_proposed > self.proposal_threshold).sum().float().item()
            totals['matched_proposals'].append(mp)
            totals['matched_true'].append((best_iou_actual > self.proposal_threshold).sum().float().item())

        return totals


def summarize_matches(out, target, iou_threshold):
    out_boxes = out['boxes']
    out_scores = out['scores']
    target_boxes = target['boxes']

    if out_boxes.shape[0] == 0 or target_boxes.shape[0] == 0:
        matched_mask = (out_scores.abs() < -1)  # all unmatched
        out_scores = out_scores
        target_scores = torch.zeros(target_boxes.shape[0])  # all unmatched
        ans = {'matched_mask': matched_mask,
               'out_scores': out_scores,
               'target_scores': target_scores}
    else:
        ious = torchvision.ops.box_iou(out_boxes, target_boxes)
        best_match = ious.max(dim=1).values
        matched_mask = best_match > iou_threshold
        target_scores = ((ious > iou_threshold).float() * out_scores.view(-1, 1)).max(dim=0).values
        ans = {'matched_mask': matched_mask, 'out_scores': out_scores,
               'target_scores': target_scores}

    assert ans['target_scores'].shape[0] == target_boxes.shape[0]
    assert ans['matched_mask'].shape[0] == ans['out_scores'].shape[0]
    return ans


def precision_at_rec(collection, rec_levels=[.1, .5]):
    mm = torch.cat([ct['matched_mask'] for ct in collection])
    pp = torch.cat([ct['out_scores'] for ct in collection])
    tpr = torch.cat([ct['target_scores'] for ct in collection])

    score_quantiles = 1. - np.array(rec_levels)
    score_thresholds = np.quantile(tpr, score_quantiles, interpolation='lower')

    res = []
    for (cutoff, rec) in zip(score_thresholds, rec_levels):
        qual = pp > cutoff
        ap = qual.float().sum()
        tp = mm[qual].float().sum()

        ttp = (tpr > cutoff).float().sum()
        res.append({'precision': (tp / ap).item(),
                    'recall': (ttp / tpr.shape[0]).item(),
                    'cutoff': cutoff})

    return res


def areas(boxes):
    w = (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
    h = (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
    return w * h


def gIoU(out_boxes, target_boxes):
    if out_boxes.shape[0] == 0:
        return (torch.tensor([]), torch.tensor([]))

    stacked = torch.stack([out_boxes, target_boxes], dim=-1)
    mincoords = stacked.min(dim=-1).values
    maxcoords = stacked.max(dim=-1).values
    intersection = torch.cat((maxcoords[..., :2], mincoords[..., 2:]), dim=-1)
    containing = torch.cat((mincoords[..., :2], maxcoords[..., 2:]), dim=-1)
    aI = areas(intersection)
    aU = areas(out_boxes) + areas(target_boxes) - aI
    assert (aU > 0).all()

    aC = areas(containing)
    assert (aC > 0).all()

    IoUs = aI / aU
    cTerm = (aC - aU) / aC
    gIoUs = IoUs - cTerm
    return (gIoUs, IoUs)


bdd_root='/big_fast_drive/orm/bdd_dataset/bdd100k/'

class BoxDetector(pl.LightningModule):
    def __init__(self, hparams):  # , ds_train, ds_val):
        super(BoxDetector, self).__init__()
        hparams['D'] = 32 * hparams['pool']
        self.hparams = hparams
        hparams2 = ndict(**hparams)
        self.hparams2 = hparams2
        hidden_channels = self.hparams2.hidden_channels

        szs = [hparams2.in_features] + [self.hparams2.hidden_channels] * self.hparams2.hidden_layers + [5]
        layers = [
            nn.AvgPool2d(kernel_size=hparams2.pool, stride=hparams2.pool, ceil_mode=True),
        ]

        lpairs = list(zip(szs[:-1], szs[1:]))
        for (i, (fin, fout)) in enumerate(lpairs):
            if i < len(lpairs) - 1:
                mod = nn.Sequential(
                    nn.Conv2d(in_channels=fin, out_channels=fout,
                              kernel_size=hparams2.ksize, padding=hparams2.ksize // 2, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU()
                )
                layers.append(mod)
            else:
                layers.append(nn.Conv2d(in_channels=fin, out_channels=fout,
                                        kernel_size=hparams2.ksize, padding=hparams2.ksize // 2))

        self.mod = nn.Sequential(*layers)

        stpath = '/big_fast_drive/bdd_activations/bdd_resnet18_mean_std.npz'
        self.act_xform = Normalizer(stpath)

        self.box_encoding = BoxEncoding(hparams2=self.hparams2)
        self.label_xform = transforms.Compose([BddBoxesT(lambda ob: ob['category'] == hparams2.object_class),
                                               self.box_encoding])

        self.ds_train = self.make_ds('train')
        self.ds_val = self.make_ds('val')
        self.loss_fn = DetLoss(hparams=hparams2)

    @staticmethod
    def _img_max_vals(bxs):
        bs = bxs.shape[0]
        mxs = bxs[:, 0].view(bs, -1).max(dim=-1).values
        return mxs

    def make_ds(self, train_or_val):
        hparams = self.hparams
        labels = BDDLabels(train_or_val, xforms=self.label_xform, root=bdd_root)
        activations = BDDActivationsZar(train_or_val, xforms=self.act_xform)
        hds = HCatDataset(activations, labels)
        return hds

    def loss(self, out, target):
        if self.hparams2.box_loss == 'mse':
            return self.loss_fn(out, target)
        elif self.hparams2.box_loss == 'gIoU':
            factor = 1. / out.shape[0]
            (out_score, out_boxes) = self.box_encoding.decode(out)
            (target_score, target_boxes) = self.box_encoding.decode(target)
            target_score = target_score.flatten(0, 1).squeeze(-1)
            out_score = out_score.flatten(0, 1).squeeze(-1)
            bceloss = self.loss_fn.bceloss(out_score, target_score) * factor
            box_mask = target_score > 0

            if box_mask.sum() > 0:
                target_boxes = target_boxes.flatten(0, 1)
                out_boxes = out_boxes.flatten(0, 1)
                (gIoUs, IoUs) = gIoU(out_boxes[box_mask], target_boxes[box_mask])
                iouloss = (1. - gIoUs).sum() * factor
            else:
                iouloss = 0.

            #             breakpoint()
            total_loss = bceloss + self.loss_fn.weight_loss_box * iouloss
            return {'loss': total_loss,
                    'loss_box': iouloss,
                    'loss_prob': bceloss}

        else:
            assert False

    def forward(self, X):
        actout = self.mod(X)
        (logits, mx, my, tw, th) = actout.unbind(dim=1)
        mx = mx.sigmoid()
        my = my.sigmoid()
        projected = torch.stack([logits, mx, my, tw, th], dim=1)
        return projected

    @torch.no_grad()
    def detect(self, Xs, **kwargs):
        '''used for inference only, includes postprocessing'''
        ypreds = self.mod(Xs)
        ans = self.box_encoding.post_process_batch(ypreds.to('cpu'),
                                                   assume_logits=True, **kwargs)
        return ans

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        data, target = batch
        out = self.forward(data)
        losses = self.loss(out, target)

        log = losses.copy()
        log['train_loss'] = log['loss']
        del log['loss']

        # losses['loss'].backward(retain_graph=True)

        return {'loss': losses['loss'], 'log': log}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_train, batch_size=self.hparams2.batch_size,
                                           num_workers=self.hparams2.num_workers, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_val, batch_size=self.hparams2.val_batch_size,
                                           num_workers=self.hparams2.num_workers, shuffle=False)

    def configure_optimizers(self):
        opt = torch.optim.SGD(params=self.parameters(), lr=self.hparams2.initial_lr)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.hparams2.milestones,
                                                     gamma=self.hparams2.gamma,
                                                     last_epoch=-1)
        return [opt], [sched]

    def validation_step(self, batch, batch_idx):
        x, ytrue = batch
        ypred = self.forward(x)
        bsize = ypred.shape[0]
        val_losses = self.loss(ypred, ytrue)
        ypred_agg = BoxDetector._img_max_vals(ypred).sigmoid()
        ytrue_agg = BoxDetector._img_max_vals(ytrue)

        (out_score, out_boxes) = self.box_encoding.decode(ypred)
        (target_score, target_boxes) = self.box_encoding.decode(ytrue)
        target_score = target_score.flatten(0, 1).squeeze(-1)
        out_score = out_score.flatten(0, 1).squeeze(-1)
        target_boxes = target_boxes.flatten(0, 1)
        out_boxes = out_boxes.flatten(0, 1)
        box_mask = target_score > 0
        (gIoUs, IoUs) = gIoU(out_boxes[box_mask], target_boxes[box_mask])

        out = self.box_encoding.post_process_batch(ypred.to('cpu'), nms_threshold=.5,
                                                   score_threshold=0., assume_logits=True)

        target = self.box_encoding.post_process_batch(ytrue.to('cpu'), nms_threshold=.5,
                                                      score_threshold=0., assume_logits=False)

        match_summaries = []
        for (o, t) in zip(out, target):
            d = summarize_matches(o, t, iou_threshold=.5)
            match_summaries.append(d)

        ret = {'val_loss': val_losses['loss'], 'y_hat': ypred_agg, 'y_true': ytrue_agg,
               'other_loss': val_losses, 'match_summaries': match_summaries, 'gIoUs': gIoUs,
               'IoUs': IoUs}

        return ret

    def validation_epoch_end(self, outputs):
        val_loss = [x['val_loss'] for x in outputs]  # single value per batch
        y_hat = [x['y_hat'] for x in outputs]  # multiple values per batch
        y_true = [x['y_true'] for x in outputs]

        match_summaries = sum([x['match_summaries'] for x in outputs], [])
        [prAt10, prAt50] = precision_at_rec(match_summaries, rec_levels=[.1, .5])

        lp = [x['other_loss']['loss_prob'] for x in outputs]
        lb = [x['other_loss']['loss_box'] for x in outputs]
        #         lsz = [ x['other_loss']['loss_sizes'] for x in outputs]
        #         lcent =  [ x['other_loss']['loss_center'] for x in outputs]

        mean_gIoU = torch.cat([x['gIoUs'] for x in outputs]).mean()
        mean_IoU = torch.cat([x['IoUs'] for x in outputs]).mean()

        loss_prob = torch.tensor(lp).float().mean()
        loss_box = torch.tensor(lb).float().mean()
        val_loss_mean = torch.stack(val_loss).mean()
        #         loss_size = torch.tensor(lsz).float().mean()
        #         loss_center = torch.tensor(lcent).float().mean()

        y_pred = torch.cat(y_hat).view(-1).to('cpu').numpy()
        y_true = torch.cat(y_true).view(-1).to('cpu').numpy()

        if not np.isnan(y_pred).any():
            AP = average_precision_score(y_true, y_pred)
        else:
            AP = np.nan

        randomAP = average_precision_score(y_true, np.random.randn(y_true.shape[0]))
        dct = {'val_loss': val_loss_mean}
        dct['log'] = {'val_frameAP': torch.tensor(AP).float(),
                      'val_randomAP': torch.tensor(randomAP).float(),
                      'val_loss': val_loss_mean,
                      'val_loss_prob': loss_prob,
                      'val_loss_box': loss_box,
                      #                       'val_loss_size':loss_size,
                      #                       'val_loss_center':loss_center,
                      'val_precision@10rec': torch.tensor(prAt10['precision']).float(),
                      'val_cutoff@10rec': torch.tensor(prAt10['cutoff']).float(),
                      'val_precision@50rec': torch.tensor(prAt50['precision']).float(),
                      'val_cutoff@50rec': torch.tensor(prAt50['cutoff']).float(),
                      'val_iou': mean_IoU,
                      'val_giou': mean_gIoU
                      }
        return dct



## test example
class MnistModel(pl.LightningModule):
    def __init__(self, hparams=dict(train_batch_size=10, lr=.1, milestones=[2, 4, 6], gamma=.5,
                      hidden_units=20, num_workers=4, val_batch_size=100,
                      train_size=10000)):

        super().__init__()
        self.hparams = hparams
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=torch.tensor(0.1307),
                                             std=torch.tensor(0.3081))
        ])

        ds_train = torchvision.datasets.MNIST(root='/nvme_drive/orm/MNIST',
                                              download=True, train=True,
                                              transform=self.transforms)

        indices = np.random.permutation(len(ds_train))[:self.hparams['train_size']]
        self.ds_train = torch.utils.data.Subset(ds_train, indices)
        self.ds_val = torchvision.datasets.MNIST(root='/nvme_drive/orm/MNIST/', download=True, train=False,
                                                 transform=self.transforms)

        num_hidden = hparams['hidden_units']
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, num_hidden, (3, 3), 1, padding=1),
                nn.BatchNorm2d(num_hidden),
                nn.ReLU(),
            ),
            nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True),
            nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden, (3, 3), 1, padding=1),
                nn.BatchNorm2d(num_hidden),
                nn.ReLU(),
            ),
            nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_hidden, 10)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def loss(self, out, target):
        return self.loss_fn(out, target)

    def forward(self, X):
        return self.model(X)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_train,
                                           batch_size=self.hparams['train_batch_size'], shuffle=True,
                                           num_workers=self.hparams['num_workers'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_val,
                                           batch_size=self.hparams['val_batch_size'], shuffle=False,
                                           num_workers=self.hparams['num_workers'])

    def configure_optimizers(self):
        opt = torch.optim.SGD(params=self.parameters(), lr=self.hparams['lr'])
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.hparams['milestones'],
                                                     gamma=self.hparams['gamma'], last_epoch=-1)
        return [opt], [sched]

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        data, target = batch
        out = self.model(data)
        loss = self.loss(out, target)
        return {'loss': loss, 'log': {'loss': loss}}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        out = self.model(data)
        losses = self.loss(out, target)
        accuracy = (out.max(dim=-1).indices == target)
        return {'val_loss': losses, 'val_acc': accuracy}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([o['val_loss'] for o in outputs]).mean()
        acc = torch.cat([o['val_acc'] for o in outputs]).float().mean()
        return {'val_loss': loss, 'log': {'val_acc': acc, 'val_loss': loss}}



import sklearn as sk
from torch.utils.data import Subset,ConcatDataset,DataLoader

def prepare_data(config):
    labs = pd.read_parquet('/nvme_drive/orm/bdd_label_counts.parquet').loc[True].iloc[:10000]
    bk = labs['bike']
    rd = labs['rider']
    bike_no_rider = ((bk > 0) & (rd < 1))
    bike_w_rider = (bk > 0) & ( bk == rd)
    neither = (bk < 1) & (rd < 1)
    pos = labs.loc[bike_no_rider]
    neg = labs.loc[~bike_no_rider]

    acts = BDDActivationsCoarse(xforms=lambda x : x.transpose(2,0,1))
    rider_box = BDDLabels('train', xforms=transforms.Compose([BddBoxesT(lambda ob: ob['category'] in ['rider']),
                                          Boxes2MaskT(d=config['d'], image_size=(1280, 720))]))
    bike_box = BDDLabels('train', xforms=transforms.Compose([BddBoxesT(lambda ob: ob['category'] in ['bike']),
                                          Boxes2MaskT(d=config['d'], image_size=(1280, 720))]))
    zero_mask = BDDLabels('train', xforms=transforms.Compose([BddBoxesT(lambda ob: False),
                                          Boxes2MaskT(d=config['d'], image_size=(1280, 720))]))

    def sel(tup):
        (a,b,r,z,bnr) = tup
        x = b if bnr else z
        return (a, (b,r,x))

    xy = HCatDataset([acts, bike_box, rider_box, zero_mask, PandasDataset(bike_no_rider)], xforms=sel)

    ds = xy
    train_ds = Subset(ds, range(5000))
    val_ds = Subset(ds,range(5000,10000))

    sub_pos = bike_no_rider[bike_no_rider].loc[:5000].sample(config['num_pos']).index - 1
    sub_partial = bike_w_rider[bike_w_rider].loc[:5000].sample(config['num_partial']).index - 1
    sub_neg = neither[neither].loc[:5000].sample(config['num_neg']).index -1

    assert (sub_pos < 5000).all() # bike alone
    assert (sub_partial < 5000).all() # bike but with rider
    assert (sub_neg < 5000).all() # negative (neither)

    balanced_train_ds = ConcatDataset([Subset(train_ds, sub_pos),
                                        Subset(train_ds, sub_partial),
                                        Subset(train_ds, sub_neg),])
    return balanced_train_ds, val_ds


class CompositeDetector(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        hparams2 = ndict(**hparams)
        self.hparams2 = hparams2
        self.composite_loss = self.hparams2.composite_loss
        self.composite_model = self.hparams2.composite_model

        if self.composite_model:
            self.mod = nn.Sequential(
                nn.Conv2d(in_channels=hparams2.in_features, out_channels=2, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=2),
                nn.Sigmoid(),
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, padding=0)
            )
        else:
            self.mod = nn.Conv2d(in_channels=hparams2.in_features,
                                 out_channels=1, kernel_size=1, padding=0)

        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean',
                                            pos_weight=torch.tensor(hparams2.pos_weight))

    def prepare_data(self):
        self.train_ds, self.val_ds = prepare_data(self.hparams)

    def loss(self, outs, targets):
        (o1, o2, o3) = outs
        (t1, t2, t3) = targets.unbind(dim=-1)

        loss3 = self.loss_fn(o3, t3)

        if self.composite_loss:
            loss1 = self.loss_fn(o1, t1)
            loss2 = self.loss_fn(o2, t2)
            loss = 2 * loss3 + loss2 + loss1
            ans = dict(loss=loss, loss1=loss1, loss2=loss2, loss3=loss3)
        else:
            loss = loss3
            ans = dict(loss=loss, loss3=loss3)

        return ans

    def forward(self, X):
        if self.composite_loss:
            o1o2 = self.mod[0](X)
            o3 = self.mod[1:](o1o2)
            o1, o2 = o1o2.unbind(dim=1)
            return o1, o2, o3.squeeze(dim=1)
        else:
            o3 = self.mod(X)
            return None, None, o3.squeeze(dim=1)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        data, target = batch
        out = self(data)
        loss_info = self.loss(out, target)
        return {'loss': loss_info['loss'], 'log': loss_info}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, batch_size=int(self.hparams2.batch_size),
                                           num_workers=self.hparams2.num_workers, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.hparams2.val_batch_size,
                                           num_workers=self.hparams2.num_workers)

    def configure_optimizers(self):
        opt = torch.optim.SGD(params=self.parameters(), lr=self.hparams2.lr)
        # torch.optim.lr_scheduler.
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.hparams2.milestones,
                                                     gamma=self.hparams2.gamma,
                                                     last_epoch=-1)

        return [opt], [sched]

    def _post_process(self, out):
        y = out[-1]
        bsize = y.shape[0]
        frame_agg = y.view((bsize, -1)).max(dim=-1).values
        return frame_agg

    def predict_proba(self, X):
        return self._post_process(self(X)).sigmoid()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        losses = self.loss(out, y)

        ypred_agg = self._post_process(out).sigmoid()
        ytrue_agg = self._post_process(y.unbind(dim=-1))

        ret = {'val_loss': losses['loss'], 'y_hat': ypred_agg, 'y_true': ytrue_agg}
        return ret

    def validation_epoch_end(self, outputs):
        val_loss = [x['val_loss'] for x in outputs]  # single value per batch
        y_hat = [x['y_hat'] for x in outputs]  # multiple values per batch
        y_true = [x['y_true'] for x in outputs]

        val_loss_mean = torch.stack(val_loss).mean()

        y_pred = torch.cat(y_hat).view(-1).to('cpu').numpy()
        y_true = torch.cat(y_true).view(-1).to('cpu').numpy()
        y_rand = np.random.randn(y_true.shape[0])

        k = self.hparams2.topk
        top_prec = top_k_prec(y_true, y_pred, k=k)
        top_prec_random = top_k_prec(y_true, y_rand, k=k)

        AP = sk.metrics.average_precision_score(y_true, y_pred)
        randomAP = sk.metrics.average_precision_score(y_true, y_rand)

        ndcg = sk.metrics.ndcg_score(y_true.reshape(1, -1), y_pred.reshape(1, -1), k=k)
        random_ndcg = sk.metrics.ndcg_score(y_true.reshape(1, -1), y_rand.reshape(1, -1), k=k)

        log = {'val_loss': val_loss_mean,
               'top_k'.format(k=k): torch.tensor(top_prec * k).float(),
               'top_k_random'.format(k=k): torch.tensor(top_prec_random * k).float(),
               'frameAP': torch.tensor(AP).float(),
               'frameAP_random': torch.tensor(randomAP).float(),
               'ndcg_score'.format(k=k): torch.tensor(ndcg).float(),
               'ndcg_score_random'.format(k=k): torch.tensor(random_ndcg).float()
               }

        return {
            'val_loss': val_loss_mean,
            'log': log
        }


def pooled_score(ytrue, ypred, scoring_function=average_precision_score):
    assert len(ytrue.shape) >= 3
    assert len(ypred.shape) >= 3

    yt = ytrue.max(axis=(1, 2))
    yp = ypred.max(axis=(1, 2))
    if isinstance(scoring_function, (list, tuple)):
        return list(map(lambda sf: sf(yt, yp), scoring_function))
    else:
        return scoring_function(yt, yp)

from sklearn.base import BaseEstimator,clone
from sklearn.linear_model import LogisticRegression

class SpatialPooledClassifier(BaseEstimator):
    def __init__(self, estimator=LogisticRegression()):
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        self.model_ = self.estimator
        X = X.reshape(-1, X.shape[-1])

        assert len(y.shape) >= 3
        if len(y.shape) > 3 and y.shape[-1] > 1:
            y = y.reshape(-1, y.shape[-1])
        else:  # len(y.shape) == 3 or y.shape[-1] == 1:
            y = y.reshape(-1)

        self.model_.fit(X, y, **kwargs)
        return self

    def predict_proba(self, X):
        orig_shape = X.shape
        X = X.reshape(-1, orig_shape[-1])
        yf = self.model_.predict_proba(X)
        yshp = yf.shape[1:]
        ys = yf.reshape(*(orig_shape[:-1] + yshp))
        return ys

## try to balance ys by downsampling
def make_balanced(X,y):
    pos = y > 0
    neg = ~pos
    positions = np.where(~pos)[0]
    n = int(pos.sum())
    ch = np.random.choice(positions,size=2*n)
    assert not y[ch].any()
    return np.concatenate([X[pos], X[ch]]), np.concatenate([y[pos], y[ch]])


from scipy.special import expit
def batch_predict_proba(lr_list, X):
    """ converts trained sklearn LRmodels to execute in batch against the same data
    :param lr_list: list of LogisticRegression models, or None
    :param X: data to run against
    :return: equivalent to stacking predict_proba for each model along dim -2
    """
    ans = np.zeros((X.shape[0], len(lr_list)))

    valids = [i for (i,lr) in enumerate(lr_list) if lr is not None]
    if len(valids) > 0:
        valid_lr = [lr_list[i] for i in valids]
        coefs = np.concatenate([lr.coef_ for lr in valid_lr],axis=0)
        icepts = np.concatenate([lr.intercept_ for lr in valid_lr],axis=0)
        pos_prob = expit(X @ coefs.transpose() + icepts)
        ans[:,valids] = pos_prob # scatter to correct pos
    return np.stack([1-ans, ans],axis=-1)

## used to test batch_predict_proba
def seq_predict_proba(lr_list, X):
    ps = []
    for lr in lr_list:
        if lr is None:
            deflt = np.zeros((X.shape[0], 2))
            deflt[:, 0] = 1.
            ps.append(deflt)
        else:
            ps.append(lr.predict_proba(X))

    v2 = np.stack(ps, axis=-2)
    return v2

class CompositeLrSkLearn(BaseEstimator):
    def __init__(self, C=1., class_weight='balanced', max_iter=1000):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter

    ## always assumes y dim
    def fit(self, X, y):
        estimator_ = LogisticRegression(C=self.C, class_weight=self.class_weight,
                                        max_iter=self.max_iter)
        self.models_ = [None] * y.shape[-1]  # , clone(estimator_), clone(estimator_)]

        for i in range(y.shape[-1]):
            yi = y[:, i]
            if yi.sum() == 0:
                continue

            self.models_[i] = clone(estimator_)
            if False:  # didn't seem to greatly help, the way we do it
                X, yi = make_balanced(X, yi)
            self.models_[i].fit(X, yi)

        return self

    def predict_proba(self, X):
        def make_default(X):
            p = .5  # .5/X.shape[0] ## just a constant small number.
            X0 = np.zeros(X.shape[0]) + p  ## still may want to multiply it
            X0 = np.stack([1 - X0, X0], axis=1)
            return X0

        ys = []
        return batch_predict_proba(self.models_, X)

    def predict(self, X):
        return self.predict_proba(X)[..., -1] > .5

    def score(self, X, ytrue, score_fun=average_precision_score):
        self.predict_proba(X)[..., -1]