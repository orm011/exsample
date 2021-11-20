import torch
import numpy as np
import os
import skvideo.io
import PIL
import pandas as pd
from collections import OrderedDict
from .dataset_tools import HCatDataset, DataFrameDataset
from .exsample_dataset import DatasetMetadata

class BlazeItVideo(torch.utils.data.Dataset):
    def __init__(self, folder, tx=None):
        self.folder = folder
        vds = [f for f in os.listdir(folder) if f.endswith('.mp4')]
        nums = sorted([int(vd.split('.')[0]) for vd in vds])
        assert nums[0] == 1
        assert nums[-1] == len(nums)

        idxpath = folder + 'sizes.npy'
        if os.path.exists(idxpath):
            sizes = np.load(idxpath)
        else:
            sizes = (np.ones(len(nums)) * 150)

        self.index = np.concatenate([[0], np.cumsum(sizes)])  # start at 0
        self.tx = (lambda x : x) if tx is None else tx

    def __len__(self):
        return self.index[-1]

    def __getitem__(self, idx):
        vnum = np.searchsorted(self.index, idx, 'right')  # if equal to value, pick position to right.
        assert idx < self.index[vnum]
        assert vnum >= 1

        fnum = idx - self.index[vnum - 1]

        vr = skvideo.io.vreader(self.folder + '{}.mp4'.format(vnum))
        for (i, f) in enumerate(vr):
            if i == fnum:
                break

        image = PIL.Image.fromarray(f)
        return self.tx(image)

def make_blazeit_ds(bzitroot, dsname):
    boxpath = '{}/filtered/{}/'.format(bzitroot, dsname)
    first_file = sorted(os.listdir(boxpath))[0]
    dsdate = '-'.join(first_file.split('-')[-3:]).split('.')[0]
    videodir = '{}/svideo/{}/{}/'.format(bzitroot, dsname, dsdate)

    boxpath = '/big_fast_drive/orm/blazeit_data/new_labels/{}-{}.csv'.format(dsname, dsdate)
    video = BlazeItVideo(videodir)
    df = pd.read_csv(boxpath)
    boxes = DataFrameDataset(df.assign(zero_based_idx=df.frame), index_var='zero_based_idx')
    return HCatDataset([video, boxes])


blazeit_dataset_names = ['venice-rialto', 'archie-day', 'taipei-hires', 'amsterdam',
                                       'venice-grand-canal', 'jackson-town-square']

def make_dms():
    blazeit_dms = []
    for ds in blazeit_dataset_names:
        folder = '/big_fast_drive/orm/blazeit_data/gdrive/Release/filtered/{}/'.format(ds)
        file = os.listdir(folder)[0]

        dm = DatasetMetadata(name=ds,
                             boxpath=folder + file,
                             track_id='ind',
                             frame_id='frame',
                             video_len=None,  # max() + 1. 11 off
                             category='object_name',
                             categories=None,
                             default_chunks=None,
                             logitpath=None,
                             logitsplit=None)

        blazeit_dms.append(dm)
    return blazeit_dms

_blazeit_dms = [ DatasetMetadata(name='venice-rialto',
                 boxpath='/big_fast_drive/orm/blazeit_data/gdrive/Release/filtered/venice-rialto/venice-rialto-2018-01-19.csv',
                 track_id='ind', frame_id='frame', category='object_name',
                 categories=None, video_len=None, default_chunks=None, logitpath=None, logitsplit=None),
 DatasetMetadata(name='taipei-hires',
                 boxpath='/big_fast_drive/orm/blazeit_data/gdrive/Release/filtered/taipei-hires/taipei-hires-2017-04-13.csv',
                 track_id='ind', frame_id='frame', category='object_name',
                 categories=None, video_len=None, default_chunks=None, logitpath=None, logitsplit=None),
 DatasetMetadata(name='venice-grand-canal',
                 boxpath='/big_fast_drive/orm/blazeit_data/gdrive/Release/filtered/venice-grand-canal/venice-grand-canal-2018-01-20.csv',
                 track_id='ind', frame_id='frame', category='object_name',
                 categories=None, video_len=None, default_chunks=None, logitpath=None, logitsplit=None),
 DatasetMetadata(name='amsterdam',
                 boxpath='/big_fast_drive/orm/blazeit_data/yolo_labels/amsterdam-2017-04-10.csv',
                 #boxpath='/big_fast_drive/orm/blazeit_data/gdrive/Release/filtered/amsterdam/amsterdam-2017-04-10.csv',
                 track_id='ind', frame_id='frame', category='object_name',
                 categories=['bicycle', 'boat', 'bus', 'car', 'person', 'truck'],
                 video_len=None, default_chunks=None, logitpath=None, logitsplit=None),
 DatasetMetadata(name='archie-day',
                 boxpath='/big_fast_drive/orm/blazeit_data/gdrive/Release/filtered/archie-day/archie-day-2018-04-11.csv',
                 track_id='ind', frame_id='frame', category='object_name',
                 categories=None, video_len=None, default_chunks=None, logitpath=None, logitsplit=None),
 DatasetMetadata(name='jackson-town-square',
                 boxpath='/big_fast_drive/orm/blazeit_data/gdrive/Release/filtered/jackson-town-square/jackson-town-square-2017-12-14.csv',
                 track_id='ind', frame_id='frame', category='object_name',
                 categories=None, video_len=None, default_chunks=None, logitpath=None, logitsplit=None)]

blazeit_meta = OrderedDict([(dm.name, dm) for dm in _blazeit_dms])

blazeit_qdetails= {'taipei-hires':('car',6), # object class, count
 'jackson-town-square':('car',5),
 'venice-rialto':('boat',7),
 'venice-grand-canal':('boat',5),
 'amsterdam':('car',4),
 'archie-day':('car',4),
}