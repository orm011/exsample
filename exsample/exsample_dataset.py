from collections import OrderedDict, namedtuple
import pandas as pd

DatasetMetadata = namedtuple('DatasetMetadata',
                             ['name',
                              'boxpath',
                              'track_id',
                              'frame_id',
                              'category',
                              'categories',
                              'video_len',
                              'default_chunks',
                              'logitpath',
                              'logitsplit'])

def read_data(path):
    if path.endswith('parquet'):
        boxes = pd.read_parquet(path)
    elif path.endswith('csv'):
        boxes = pd.read_csv(path)
    else:
        assert False

    return boxes

_dms = [
    DatasetMetadata(name='dashcam_unified_ng',
                    boxpath='/nvme_drive/orm/results10//csv/dashcam_unified_ng.track_ids.clean.parquet',
                    track_id='tid',
                    frame_id='frame_idx',
                    video_len=1164226,
                    category='category',
                    categories=[
                        'person',
                        'bicycle',
                        # 'car', # too common
                        # 'motorcycle', # det. not accurate
                        'bus',
                        'truck',
                        'traffic light',
                        'fire hydrant',
                        'stop sign',
                        # 'parking meter', # det. not accurate
                    ],
                    default_chunks=('/nvme_drive/orm/dash_metadata_by_frame.parquet', 'chunk_id'),
                    logitpath='/nvme_drive/orm/dashcam_logits_secon_version.parquet',
                    logitsplit='subset'),
    DatasetMetadata(name='bdd1k',
                    boxpath='/nvme_drive/orm/results10//csv/bdd1k.track_ids.clean.parquet',
                    track_id='tid',
                    frame_id='frame_idx',
                    video_len=389662,  # 1/3 frames of 1k ~40 second clips.
                    category='category',
                    categories=['person',
                                'rider',
                                # 'car', # too common
                                'bus',
                                'truck',
                                'bike',
                                'motor',
                                'traffic light',
                                'traffic sign', ],
                    default_chunks=('/nvme_drive/orm/bdd_metadata_by_frame.parquet', 'clip_id'),
                    logitpath='/nvme_drive/orm/car_stuff/vroom_nbs/bdd_logits_v0.parquet',
                    logitsplit='subset'),
]