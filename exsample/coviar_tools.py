import new_coviar
import numpy as np
import pandas as pd
import os
from . import avutils



def get_metadata(path):
    rows = [
        'gop_number',
        'frame_number',
        'byte_offset',
        'packet_size',
        'keyframe',
        'frame_pos', ]

    met = new_coviar.extract_metadata(path)

    return pd.DataFrame(dict(zip(rows, met))).set_index('frame_number')


class CoviarVideo:
    def __init__(self, path):
        self.path = path
        mpath = self.path + '.meta.parquet'

        if os.path.exists(mpath):
            self.metadata = pd.read_parquet(mpath)
        else:
            mt = get_metadata(path)
            mt.to_parquet(mpath)
            self.metadata = mt

        self.gop_offsets = self.metadata.groupby('gop_number').byte_offset.min()
        example_frame = new_coviar.load(self.path, 0, 0, 0, False, 0)
        self.shape = tuple([self.metadata.shape[0]] + list(example_frame.shape))
        self.view = avutils.VideoFrameViz(self, input_mode='bgr')

    def __len__(self):
        return self.shape[0]

    def __fetch_individual(self, single_pos):
        tup = self.metadata.iloc[single_pos]
        assert tup.name == single_pos
        gop_offset = self.gop_offsets.loc[tup.gop_number]
        frm = new_coviar.load(self.path, tup.gop_number, tup.frame_pos, 0, False, gop_offset)
        return frm

    def __getitem__(self, idx):
        if np.issubdtype(type(idx), np.integer):
            idx = int(idx)
            if idx < 0:
                idx = self.shape[0] + idx

            return self.__fetch_individual(idx)
        elif isinstance(idx, slice):
            vals = list(range(*idx.indices(self.shape[0])))
            ans = []
            for v in vals:
                fr = self.__fetch_individual(v)
                ans.append(fr)

            return np.stack(ans)