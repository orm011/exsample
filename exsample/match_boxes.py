import scipy
import scipy.optimize
import torch
import numpy as np
import torchvision.ops
import pyroaring as pr
import pandas as pd
from collections import defaultdict
from tqdm.auto import tqdm

def nms(new_dets, iou_threshold, score='confidence'):
    boxes1 = torch.from_numpy(new_dets[['xmin', 'ymin', 'xmax', 'ymax']].values).float()
    scores1 = torch.from_numpy(new_dets[[score]].values).float()
    keep_idcs = torchvision.ops.nms(boxes1, scores1, iou_threshold=iou_threshold).numpy()

    # all false
    keep_mask = np.ones(new_dets.shape[0]) < 0
    assert not keep_mask.any()
    keep_mask[keep_idcs] = True
    suppress_idcs = np.where(~keep_mask)

    # supressed boxes dont get a tid and dont get added to track set
    dets_suppressed = new_dets.iloc[suppress_idcs]
    new_dets = new_dets.iloc[keep_idcs]
    return new_dets, dets_suppressed

def match_boxes(boxes, frame_idx='frame_idx', category='category',
                xmin='xmin', ymin='ymin', xmax='xmax', ymax='ymax',
                confidence='confidence', start_score_min=.99,
                match_iou_min=.01, nms_iou_max=.2, grace_period_frames=30):
    print(start_score_min)
    orig_index = boxes.index
    assert boxes.index.is_unique
    boxes = boxes.assign(box_id=np.arange(boxes.shape[0]))
    boxes = boxes.assign(category=boxes[category],
                         frame_idx=boxes[frame_idx],
                         xmin=boxes[xmin], ymin=boxes[ymin],
                         xmax=boxes[xmax], ymax=boxes[ymax],
                         confidence=boxes[confidence])

    boxes = boxes[['box_id', 'category', 'frame_idx', 'xmin',
             'ymin', 'xmax', 'ymax', 'confidence']]  # avoid extra cols...

    tids = boxes.box_id.values.copy()  # each box is initially its own track, until it has been matched
    suppressed = np.ones_like(tids) < 0
    active_tracks = pr.BitMap()  # set of latest box_id for each of the active tracks (pointers)
    ## note:
    ##  ba.iloc[active_tracks] will give all box information for the currently active tracks
    ##  tids[active_tracks] gives the corresponding track information.

    GRACE_PERIOD = grace_period_frames
    IOU_MIN = match_iou_min
    n_unique = boxes.frame_idx.unique().shape[0]
    empty_dets = boxes.iloc[:0]

    for (frame_idx, detections) in tqdm(boxes.groupby('frame_idx'), total=n_unique):
        last_frames = boxes.frame_idx.values[active_tracks]
        expired = (last_frames + GRACE_PERIOD <= frame_idx)
        expired_tracks = boxes.box_id.values[active_tracks][expired]
        active_tracks.difference_update(pr.FrozenBitMap(expired_tracks))
        active_dets = boxes.iloc[active_tracks]

        det_dict = defaultdict(lambda: empty_dets,
                               list(active_dets.groupby('category')))
        for (cat, new_dets) in detections.groupby('category'):
            active_dets = det_dict[cat]
            new_box_ids = new_dets.box_id.values
            old_box_ids = active_dets.box_id.values

            boxes1 = torch.from_numpy(new_dets[['xmin', 'ymin', 'xmax', 'ymax']].values).float()
            boxes2 = torch.from_numpy(active_dets[['xmin', 'ymin', 'xmax', 'ymax']].values).float()
            ious = torchvision.ops.box_iou(boxes1, boxes2).numpy()

            ious[ious < IOU_MIN] = 0.
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(-ious)
            matched = ious[row_ind, col_ind] > 0
            new_matched = row_ind[matched]
            old_matched = col_ind[matched]

            ## to avoid fragmentation:
            # we favor new boxes with better iou to existing matches
            # than boxes with better score in the same frame.

            # we only add new boxes that overlap little with boxes in matched tracks
            # apply nms but give matched boxes large scores.
            synth_score = new_dets.confidence.values.copy()
            synth_score[new_matched] = 1.5
            new_keep, new_suppress = nms(new_dets.assign(synth_score=synth_score),
                                         score='synth_score', iou_threshold=nms_iou_max)

            # new_keep = new_keep[new_keep.confidence > start_threshold]
            suppressed[new_keep[new_keep.confidence <= start_score_min].box_id.values] = True
            suppressed[new_suppress.box_id.values] = True
            suppressed[new_box_ids[new_matched]] = False

            # keep all new things with no overlap with preference based on matched,score
            active_tracks.update(new_keep[new_keep.confidence > start_score_min].box_id.values)

            # keep all matched things (in case they overlapped with other matched things)
            active_tracks.update(new_box_ids[new_matched])

            # tids get inherited + old boxes get removed.
            tids[new_box_ids[new_matched]] = tids[old_box_ids[old_matched]]
            active_tracks.difference_update(pr.FrozenBitMap(old_box_ids[old_matched]))

    # map tids back to original index
    return pd.DataFrame(dict(tids=orig_index[tids], suppressed=suppressed), index=orig_index)

def match_boxes_pass(boxes, kwargs):
    return match_boxes(boxes, **kwargs)

def parallel_match_boxes(boxes, num_workers, partition_by, **kwargs):
    import multiprocessing
    data = [gp for (_, gp) in list(boxes.groupby(partition_by))]

    num_workers = min(num_workers, len(data))
    with multiprocessing.Pool(num_workers) as p:
        tidlist = p.starmap(match_boxes_pass, zip(data, [kwargs for _ in data]))

    tids = pd.concat(tidlist)
    return tids