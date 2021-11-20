import shapely.geometry as geom
import cv2
from collections import defaultdict
import shapely
import pandas as pd
import PIL
import numpy as np

def mask2poly(mask, epsilon=1., min_area=10.):
    from shapely.geometry import MultiPolygon, Polygon
    # https://michhar.github.io/masks_to_polygons_and_back/
    """Convert a mask ndarray (binarized image) to Multipolygons"""
    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(mask,
                                           cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)

    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            # Approximate contours for a smaller polygon array to save on memory
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area]).buffer(0).simplify(epsilon)
            all_polygons.append(poly)
    all_polygons = MultiPolygon(all_polygons)
    return all_polygons


def box2poly(bx):
    # minx,miny,maxx,maxy=bx
    return geom.box(*bx)


def ser_gdf(df):
    from shapely.wkb import dumps
    sdf = df
    sdf = sdf.assign(box=df.box.apply(dumps))
    if 'polymask' in df.columns:
        sdf = sdf.assign(polymask=df['polymask'].apply(dumps))
    return sdf


def des_gdf(serial_df):
    from shapely.wkb import loads
    df = serial_df
    df = df.assign(box=serial_df.box.apply(loads))

    if 'polymask' in df.columns:
        df = df.assign(polymask=serial_df['polymask'].apply(loads))

    return df


def process_idx(frame_idx, vd, coco_demo):
    im = vd[frame_idx] #PIL.Image.fromarray(dec.retrieve([frame_idx])[0])
    cvim = np.array(im)[..., [2, 1, 0]]
    predictions = coco_demo.compute_prediction(cvim)

    scores = predictions.get_field('scores')
    labels = predictions.get_field('labels')
    masks = predictions.get_field('mask')
    boxes = predictions.bbox
    polymasks = [mask2poly(m.squeeze().numpy()) for m in masks]
    polybox = [box2poly(b) for b in boxes]

    df = pd.DataFrame(
        {'frame_idx': frame_idx, 'scores': scores, 'labels': labels, 'polymask': polymasks, 'box': polybox})
    df = df.assign(frame_idx=frame_idx)
    return masks, df


def show_masks(df, vd, stride=10, limit=10):
    import geopandas as gpd


    df = df[(df.frame_idx - df.frame_idx.min()) % stride == 0]
    idxs = df.frame_idx.unique()[:limit]
    f, axes = plt.subplots(idxs.shape[0], figsize=(3 * 3 * idxs.shape[0], 4 * 3))
    for ax, (frame_idx, rows), i in zip(axes, df.groupby('frame_idx'), range(limit)):
        frm = vd[frame_idx]
        ax.imshow(frm)
        gs = gpd.GeoSeries(rows.box)
        gs.boundary.plot(alpha=.8, ax=ax, color='r')

        if 'polymask' in rows.columns:
            mask = gpd.GeoSeries(rows.polymask)
            mask.plot(alpha=.2, ax=ax, color='r')