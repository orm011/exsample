from __future__ import print_function
import PIL
import PIL.ImageDraw
import pandas as pd
import numpy as np
from collections import namedtuple
import PIL
import PIL.ImageDraw
import PIL.Image
import geopandas as gpd
import shapely.geometry as geom
import sys


def draw_boxes(pil_img, boxes_df, labels=[], width=5, color=None, relative=None):
    # from object_detection.utils import visualization_utils as vis_util
    pil_img = pil_img.copy()

    if type(boxes_df) is pd.DataFrame:
        iterator = boxes_df.itertuples()
        has_color_column = 'color' in boxes_df.columns.values

    else:
        iterator = boxes_df
        has_color_column = False

    draw = PIL.ImageDraw.Draw(pil_img)
    for tup in iterator:
        if 'xmin' in boxes_df.columns:
            pil_box = [tup.xmin, tup.ymin, tup.xmax, tup.ymax]
        else:
            pil_box = [tup.left, tup.top, tup.right, tup.bottom]

        draw.rectangle(pil_box, outline=color if color is not None else 'Orange', width=width)

        # this didn't work: shape.close() is expected...
        # [minx,miny,maxx,maxy] = pil_box
        # b = geom.box(minx=minx,miny=miny,maxx=maxx,maxy=maxy)
        # draw.shape(b)
    return pil_img


def show_track(detections, vd, W=3, rescale_factor=1., limit=6, width=5, color='Orange'):
    box1 = pd.DataFrame(detections)
    s = box1.frame_idx.min()
    e = box1.frame_idx.max()
    step = (e - s) // limit
    step = max(1, step)
    idxs = np.arange(s, e, step)
    box1 = box1[box1.frame_idx.isin(idxs) | (box1.frame_idx == box1.frame_idx.max())]
    box1 = box1.assign(left=box1.minx, right=box1.maxx, top=box1.miny, bottom=box1.maxy)
    box2 = box1[['left', 'right', 'top', 'bottom']] * rescale_factor
    # boxes were computed on larger video than we are displaying on.
    # one of the videos is substantially smaller
    box2['frame_idx'] = box1['frame_idx']
    #  if frame_idx between (542326,549796), use rescale factor .533, else use .8
    print('got here')
    ims = map(PIL.Image.fromarray, vd[box2.frame_idx.values])
    drawn_ims = [draw_boxes(im, box2.iloc[i:i + 1], width=width, color=color) for (i, im) in enumerate(ims)]

    carrs = []
    h = int(len(drawn_ims) // W)
    rem = len(drawn_ims) % W
    h = h + int((rem > 0))

    arr0 = np.array(drawn_ims[0])
    pad = np.zeros_like(arr0)

    print('got here 2')
    for i in range(h):
        tarr = drawn_ims[W * i:W * (i + 1)]
        npad = (W - rem) % W

        if len(carrs) > 0 and npad > 0 and i == h - 1:
            tarr = tarr + npad * [pad]

        carr = np.concatenate(tarr, axis=1)
        carrs.append(carr)

    carr = np.concatenate(carrs, axis=0)
    return PIL.Image.fromarray(carr)