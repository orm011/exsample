from builtins import object
import pandas as pd
from IPython.core.display import HTML
from collections import namedtuple
import torch
import torch.utils.data
#import torch.utils.data.Dataset
import hwang
import PIL, PIL.Image, PIL.ImageDraw
import numpy as np
import cv2

VideoMeta = namedtuple('VideoMeta', 'path valid fps height width frame_count duration time_base rotate')
import json
import subprocess
import fractions

def get_ffmpeg_metadata(video_path):
    """this should run in constant time (the metadata is not verified)"""
    metadata_cmd = "ffprobe -i {} -print_format json -loglevel fatal -show_streams -select_streams v"
    try:
        output = subprocess.check_output(metadata_cmd.format(video_path).split(" "), stderr=open('/dev/null'))
        valid = True
    except:
        valid = False

    if valid:
        res = json.loads(output)
    else:
        res = None
    return valid, res



def get_video_meta(filename, stream_no=0):
    """return meta info"""
    valid, fmeta = get_ffmpeg_metadata(filename)
    if not valid:
        return None
        # return VideoMeta(**dict(path=filename, valid=valid))

    stream = fmeta['streams'][stream_no]
    h = stream['height']
    w = stream['width']
    nb_frames = int(stream['nb_frames'])
    duration_s = float(stream['duration'])
    time_base = fractions.Fraction(stream['time_base'])
    frame_rate = float(fractions.Fraction(stream['avg_frame_rate']))

    if 'rotate' in list(stream['tags'].keys()):
        rotate = int(stream['tags']['rotate'])
    else:
        rotate = None

    return VideoMeta(path=filename,
                     valid=valid,
                     fps=frame_rate,
                     height=h,
                     width=w,
                     frame_count=nb_frames,
                     duration=duration_s,
                     time_base=time_base,
                     rotate=rotate)


def show_vid(url):
    from IPython.core.display import HTML
    template = '<video width="640" height="480" src="{url}" controls />'
    return HTML(template.format(url=url))


def read_ffprobe_json(jsonfilename):
    """reads the json generated from an ffprobe call into a dataframe.
        handles some annoyances in the ffprobe output format.

        for example, the output of something like:

        ffprobe  -select_streams 0 \
                        -show_entries \
                        frame=pict_type,coded_picture_number,pkt_dts,pkt_size,pkt_pos,key_frame \
                        -print_format json   \
                        /nvme_drive/full_rendered/sam_trip19_frameno_builtin_ffmpeg_preset_slow.mp4\
                        > /tmp/ffprobe_output.json
    """
    assert jsonfilename.endswith('.json')
    raw_js = json.load(open(jsonfilename))
    ff_df = pd.DataFrame(raw_js['frames'])

    if 'pkt_pos' in ff_df.columns:
        ff_df['pkt_pos'] = ff_df.pkt_pos.astype('int64')

    if 'pkt_size' in ff_df.columns:
        ff_df['pkt_size'] = ff_df.pkt_size.astype('int64')

    return ff_df


import cv2


def show_frames(frames, margin=1, W=5, target_width=900, input_mode='rgb'):
    if len(frames.shape) == 3:
        frames = np.expand_dims(frames, 0)

    assert frames.shape[0] > 0
    frames = frames.copy()

    (n, h, w, c) = frames.shape
    # want tw = W*frame.W*fx
    # fx = tw/W/frame.W
    if n < W:
        W = n

    fx = target_width / W / w
    if fx > 1:  # don't magnify
        fx = 1.

    fy = fx

    frames = np.stack(list(map(lambda fr: cv2.resize(fr, None, None, fx, fy, cv2.INTER_LINEAR), frames)))
    (_, hp, wp, c) = frames.shape
    assert wp * W <= target_width

    frames[:, -margin:, :, :] = 0
    frames[:, :, -margin:, :] = 0

    rows = []
    for (i, r) in enumerate(range(0, len(frames), W)):
        row = frames[r:r + W]
        n, h, w, c = row.shape
        row = row.transpose((1, 0, 2, 3)).reshape(h, w * n, c)
        rows.append(row)

    H = len(rows)
    rem = len(frames) % W
    if H > 1 and rem > 0:
        pad = np.zeros_like(frames[0])
        last_row = np.concatenate([rows[-1]] + [pad] * (W - rem), axis=1)
        rows[-1] = last_row

    carr = np.concatenate(rows, axis=0)

    if input_mode == 'bgr':
        carr = carr[..., [2, 1, 0]]

    def xymapper(x, y):
        # get i,j position
        I = y // hp
        J = x // wp

        # map i,j to back to frame idx in input
        N = I * W + J
        if N > (frames.shape[0] - 1):
            return -1
        else:
            return int(N)

    return PIL.Image.fromarray(carr), xymapper


import PIL.Image
import PIL.ImageFont
import PIL.ImageDraw

def mark_number(frmid, frm):
    asim = PIL.Image.fromarray(frm)
    imdraw = PIL.ImageDraw.Draw(asim, 'RGBA')
    fnt = PIL.ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', size=40)
    imdraw.text((20, frm.shape[0] - 40), str(frmid), font=fnt, color=(255, 255, 255, 120))
    return np.array(asim)

def mask_boolean(series):
    def fun(frmid, frm):
        asim = PIL.Image.fromarray(frm)
        imdraw = PIL.ImageDraw.Draw(asim, 'RGBA')
        if (series.loc[frmid] > 0):
            imdraw.rectangle([0, 0, frm.shape[1], frm.shape[0]], fill=(255, 255, 255, 150), outline=None,
                                     width=0)
        return np.array(asim)

    return fun

class OldVideoFrameViz(object):
    def __init__(self, vd, callbacks=[mark_number], interactive=False, **kwparams):
        self.vd = vd
        self.callbacks = callbacks
        self.params = kwparams
        self.cached_slice = None
        self.cached_data = None

    def options(self, callbacks=[mark_number], interactive=False, **kwargs):
        return VideoFrameViz(self.vd, callbacks, interactive, **{**self.params, **kwargs})

    def __getitem__(self, sl):
        if isinstance(sl, slice) and self.cached_slice == sl:
            frms = self.cached_data
        else:
            frms = self.vd.__getitem__(sl)
            self.cached_slice = sl
            self.cached_data = frms

        idcs = list(range(*sl.indices(self.vd.shape[0])))
        assert (frms.shape[0] == len(idcs))

        out_frms = []
        for idx, frm in zip(idcs, frms):
            tmp = frm
            for cb in self.callbacks:
                tmp = cb(idx, frm)

            out_frms.append(tmp)

        frms = np.stack(out_frms)
        im, xymapper = show_frames(frms, **self.params)
        return im, xymapper

class VideoFrameViz(object):
    def __init__(self, vd, callbacks=[], interactive=False, **kwparams):
        self.vd = vd
        self.callbacks = callbacks
        self.params = kwparams
        self.cached_slice = None
        self.cached_data = None

    def options(self, callbacks=[], interactive=False, **kwargs):
        return VideoFrameViz(self.vd, callbacks, interactive, **{**self.params, **kwargs})

    def __getitem__(self, sl):
        frms = self.vd.__getitem__(sl)
        #         assert (frms.shape[0] == len(idcs))
        if isinstance(sl, slice):
            idcs = list(range(*sl.indices(self.vd.shape[0])))
        else:
            idcs = list(sl)

        out_frms = []
        for idx, frm in zip(idcs, frms):
            tmp = frm
            for cb in self.callbacks:
                tmp = cb(idx, frm)

            out_frms.append(tmp)

        frms = np.stack(out_frms)
        im, xymapper = show_frames(frms, **self.params)
        return im

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, mp4path, transform=None,  **hopts):
        self.mp4path = mp4path
        self.vid_idx = hwang.index_video(self.mp4path)
        self.h = int(self.vid_idx.frame_height())
        self.w = int(self.vid_idx.frame_width())
        self.len = int(self.vid_idx.frames())
        self.hopts = hopts
        self._dec = None #MyDecoder(self.mp4path, self.vid_idx, **self.hopts)
        # self._worker_id = None #torch.utils.data.get_worker_info()
        self.transform = transform if transform is not None else lambda x: x
        self.shape = [int(self.vid_idx.frames()), self.h, self.w, 3]
        self.view = VideoFrameViz(self, input_mode='rgb')
        self.base_range = range(self.len)

    def __len__(self):
        return len(self.base_range)

    def __getitem__(self, idx):
        # info = torch.utils.data.get_worker_info()
        if self._dec is None:
            self._dec = Decoder2(self.mp4path, self.vid_idx, **self.hopts)
            # self._dec_worker = info.id

        if isinstance(idx, range):
            actual_idxs = [self.base_range[i] for i in idx]
        elif isinstance(idx, (list,np.ndarray)):
            actual_idxs = idx
        else: # eg. slice() or int like
            actual_idxs = self.base_range.__getitem__(idx)

        if isinstance(actual_idxs, int):
            return self.transform(self._dec.retrieve([actual_idxs])[0])
        elif isinstance(actual_idxs, (range,list,np.ndarray)):
            idx = list(actual_idxs)
            pos = np.array(idx)
            perm = np.argsort(pos)

            ## the library seems to have issues with
            ## out-of-order frame numbers in the same call
            inv_perm = np.argsort(perm)
            sorted_pos = pos[perm]
            sorted_frms = list(zip(sorted_pos, self._dec.retrieve(sorted_pos)))

            tmps = [sorted_frms[i] for i in inv_perm]
            id2 = np.array([i for (i,_) in tmps])
            frms = [self.transform(fr) for (i,fr) in tmps]
            assert (id2 == idx).all()

            if torch.is_tensor(frms[0]):
                return torch.stack(frms)
            elif isinstance(frms[0], np.ndarray):
                return np.stack(frms)
            else:
                return frms
        else:
            assert False


class VideoFile(object):
    def __init__(self, path):
        self.path = path
        self.meta =  get_video_meta(path)
        self.packet_meta = None

    def _check_frame_no(self, frame_no):
        assert frame_no >= 0
        assert frame_no < self.meta.frame_count, 'frame_no beyond range: got: {fno} video_length: {vl} '.format(fno=frame_no, vl=self.meta.frame_count)

    def get_video_meta(self):
        return self.meta

    def to_html(self, time_seconds=None, server='http://localhost:8001'):
        template = '<video width="640" height="480" src="{server}/{path}{time}" controls />'
        time = "#t={time}".format(time=time_seconds) if time_seconds is not None else ''
        return HTML(template.format(server=server, path=self.path, time=time))

    def show_video(self, time_seconds=None, frame_no=None):
        self._check_frame_no(frame_no)
        if frame_no is not None:
            time_seconds=frame_no/1.0/self.meta.fps

        return self.to_html(time_seconds=time_seconds)