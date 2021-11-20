import os
# needed in supercloud machines
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import ray
import torch
from ray.util import ActorPool

from torch.utils.data import DataLoader
from  tqdm.auto import tqdm
import pandas as pd
import skvideo
import skvideo.io
from torchvision import transforms as T
import pathlib

tx = T.Compose([T.ToPILImage(), T.Resize(800), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class SkVideoClip(torch.utils.data.IterableDataset):
    def __init__(self, path, tx=None):
        self.path = path
        self.tx = (lambda x: x) if tx is None else tx

    def __iter__(self):
        vr = skvideo.io.vreader(self.path)
        for frame in vr:
            yield self.tx(frame)


def run_model(model, im=None, transformed_img=None, tx=None):
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
        transformed_img = tx(im).unsqueeze(0)

    outputs = model(transformed_img)
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


@ray.remote(num_gpus=.32, num_cpus=4)
class GPUModel(object):
    def __init__(self, checkpoint_path):
        torch.set_grad_enabled(False)
        mod = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        mod.load_state_dict(checkpoint['model'])
        self.dev = f'cuda:0'  # it always appears as 0
        self.mod = mod.eval().to(self.dev)

    def run_on_clip(self, loc, date, vidname):
        ofile = vidname.as_posix().replace('.mp4', '_dets.parquet')
        if os.path.exists(ofile):
            return

        ds = SkVideoClip(vidname.as_posix(), tx=tx)
        dl = DataLoader(ds, num_workers=1, pin_memory=True)
        dets = []
        for (i, frm) in enumerate(dl):
            frm = frm.to(self.dev)
            det = run_model(self.mod, transformed_img=frm)
            det = det.assign(dataset=loc, date=date, local_frame_idx=i,
                             path=vidname.as_posix())
            det = det[det.confidence > .5]
            dets.append(det)

        all_dets = pd.concat(dets, ignore_index=True)
        all_dets = all_dets.assign(num_frames=i)
        all_dets.to_parquet(ofile)

def process_clips(location):
    home = '/home/gridsan/omoll/'
    vidroot = f'{home}/gdrive/Release/svideo/'
    actors = [GPUModel.remote(f'/home/gridsan/omoll/models/finetune_{location}/checkpoint.pth') for _ in range(6)]
    pool = ActorPool(actors)

    cls = []
    path = pathlib.Path(f'{vidroot}/{location}/')
    for date in tqdm(os.listdir(path), leave=False):
        if date.endswith('zip'):
            continue
        vp = path / date
        paths = list(vp.glob('*mp4'))
        r = pool.map(lambda ac, vidname: ac.run_on_clip.remote(location, date, vidname), paths)
        cls.append(r)

    return [list(m) for m in cls]

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', help='location')
    args = parser.parse_args()
    ray.init()
    process_clips(location=args.location)