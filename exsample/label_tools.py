import pycocotools
import pandas as pd
import json
import tempfile
from pycocotools import COCO

def make_COCO(coco_labels):
    with tempfile.NamedTemporaryFile('w+') as tempf:
        json.dump(coco_labels,tempf.file)
        tempf.file.flush()
        tempf.file.close()
        coco = COCO(tempf.name)
        return coco


DETR_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def dfstyle2bddstyle(all_preds, threshold,
                     score_col='confidence', category_col='category', path_col='path',
                     video_name='video_name', id_col=None):
    bdd_label = []
    all_preds = all_preds.assign(score=all_preds[score_col],
                                 category=all_preds[category_col],
                                 path=all_preds[path_col])

    if id_col is not None:
        all_preds = all_preds.assign(id_col=all_preds[id_col])
    else:
        all_preds = all_preds.assign(id_col=range(all_preds.shape[0]))

    for ((path, frmidx), boxes) in all_preds.groupby([path_col, 'frame_idx']):
        det_dict = {}
        det_dict["name"] = path
        det_dict["url"] = 'http://clauslocal:8000/{}'.format(path)
        det_dict["attributes"] = {"weather": "undefined",
                                  "scene": "undefined",
                                  "timeofday": "undefined"}
        det_dict['height'] = int(boxes.height.max())
        det_dict['width'] = int(boxes.width.max())
        det_dict["labels"] = []
        det_dict['video_name'] = boxes.video_name.iloc[0]

        for (i, ann) in enumerate(boxes.itertuples()):
            if ann.confidence < threshold:
                continue

            label = {"id": int(ann.id_col),
                     "category": ann.category,
                     "manualShape": False,
                     "manualAttributes": False,
                     "attributes": {},
                     "score": ann.score,
                     "box2d": {
                         "x1": ann.xmin,
                         "y1": ann.ymin,
                         "x2": ann.xmax,
                         "y2": ann.ymax,
                     }}

            det_dict["labels"].append(label)
        bdd_label.append(det_dict)

    return bdd_label

from tqdm.auto import tqdm

def bdd2coco_detection(labeled_images, ordered_class_list):
    attr_dict = dict()
    categories = []
    images = list()
    annotations = list()

    for (i, c) in enumerate(ordered_class_list):
        categories.append({"supercategory": "none", "id": i, "name": c})

    attr_dict["categories"] = categories
    id_dict = {i['name']: i['id'] for i in attr_dict['categories']}

    counter = 0
    lab_counter = 0
    for i in tqdm(labeled_images):
        counter += 1
        image = dict()
        image['file_name'] = i['name'].split('/')[-1]
        image['height'] = int(i['height'])
        image['width'] = int(i['width'])
        image['id'] = counter

        empty_image = True

        for label in i['labels']:
            annotation = dict()
            if label['category'] in id_dict.keys():
                empty_image = False
                annotation["iscrowd"] = 0
                annotation["image_id"] = image['id']
                x1 = label['box2d']['x1']
                y1 = label['box2d']['y1']
                x2 = label['box2d']['x2']
                y2 = label['box2d']['y2']
                annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]
                annotation['area'] = float((x2 - x1) * (y2 - y1))
                annotation['category_id'] = id_dict[label['category']]
                annotation['ignore'] = 0
                annotation['id'] = lab_counter
                lab_counter+=1
                annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]

                if 'score' in label.keys():
                    annotation['score'] = label['score']
                annotations.append(annotation)

        if empty_image:
            continue

        images.append(image)

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"
    return attr_dict

import torch
from .dataset_tools import  std_transform

def run_model(model, im=None, transformed_img=None):
    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    if transformed_img is None:
        transformed_img = std_transform(im).unsqueeze(0)

    outputs = model(transformed_img.to('cuda:0'))
    outputs = {k: v.to('cpu') for k, v in outputs.items()}
    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values >= 0

    # convert boxes from [0; 1] to image scales
    h, w = transformed_img.size()[-2:]
    kbboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], (w, h))
    kprobas = probas[keep]
    scs, cls = kprobas.max(dim=1)

    CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    class_names = [CLASSES[cl] for cl in cls]

    bdata = pd.DataFrame({'xmin': kbboxes_scaled.numpy()[:, 0],
                          'ymin': kbboxes_scaled.numpy()[:, 1],
                          'xmax': kbboxes_scaled.numpy()[:, 2],
                          'ymax': kbboxes_scaled.numpy()[:, 3],
                          'category': class_names,
                          'category_id': cls.numpy(),
                          'confidence': scs.numpy()})

    bdata = bdata.assign(height=h, width=w)
    return bdata