import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
import numpy as np
import PIL
import pandas as pd

from detectron2.utils.visualizer import ColorMode


def _create_text_labels(classes, scores, class_names, instance_ids=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

    Returns:
        list[str] or None
    """

    if instance_ids is None:
        instance_ids = [None for _ in len(scores)]

    labels = None
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}% id:{}".format(s * 100, int(i)) for (s, i) in zip(scores, instance_ids)]
        else:
            labels = ["{} {:.0f}% id:{}".format(l, s * 100, int(i)) for (l, s, i) in zip(labels, scores, instance_ids)]
    return labels


class MyVisualizer(Visualizer):
    def draw_instance_predictions(self, predictions, colors=None, instance_ids=None):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None), instance_ids)
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            #             colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            assert predictions.has("pred_masks"), "ColorMode.IMAGE_BW requires segmentations"
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )

        return self.output


class Detectron2Setup(object):
    def __init__(self, config_name=None, cfg=None):
        """ example config_names:
            'COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml'
            'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
        """
        if cfg is None:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(config_name))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name)

        self.cfg = cfg
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5  # set threshold for this model
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = .5
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        # self.predictor = DefaultPredictor(cfg)

    def apply(self, im):
        outputs = self.predictor(np.array(im)[..., ::-1])
        return outputs

    def view(self, im, outputs, colors=None, instance_ids=None):
        v = MyVisualizer(np.array(im)[..., ::-1], self.meta, scale=1)

        if 'panoptic_seg' in outputs:
            panoptic_seg, segments_info = outputs['panoptic_seg']
            vi = v.draw_panoptic_seg_predictions(panoptic_seg.to('cpu'), segments_info)
        elif 'instances' in outputs:
            vi = v.draw_instance_predictions(outputs['instances'].to('cpu'), colors=colors, instance_ids=instance_ids)

        pim = PIL.Image.fromarray(vi.get_image()[..., ::-1])
        return pim


def detectron2df(lst, frame_ids=None):
    records = []
    if frame_ids is None:
        frame_ids = range(len(lst))

    assert len(frame_ids) == len(lst)

    for (i, f) in zip(frame_ids, lst):
        fs = f['instances'].get_fields()
        boxes = fs['pred_boxes'].tensor.numpy()
        scores = fs['scores'].numpy()
        categories = fs['pred_classes'].numpy()
        for (b, s, c) in zip(boxes, scores, categories):
            nrec = {}
            nrec['frame_idx'] = i
            nrec['label'] = c
            nrec['score'] = s
            nrec['minx'] = b[0]
            nrec['miny'] = b[1]
            nrec['maxx'] = b[2]
            nrec['maxy'] = b[3]
            records.append(nrec)

    return pd.DataFrame(records)