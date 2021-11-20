# trainer.py
from collections import Counter
import os
import sys
import time
import ray
import pandas as pd
from .exsample_sampler import *
from .exsample_benchmark import *

import argparse
parser = argparse.ArgumentParser(description='run sampling sim in parallel')
parser.add_argument('--redis-password', dest='redis_password', type=str)
parser.add_argument('--num_cpus', dest='num_cpus', type=int)
parser.add_argument('--reps', dest='reps', type=int)
parser.add_argument('--threshold', dest='threshold', type=float)

args = parser.parse_args()

ray.init(address=os.environ["ip_head"], redis_password=args.redis_password)

print("Nodes in the Ray cluster:")
print(ray.nodes())

root = '/home/gridsan/omoll/data/exsample_data/'
#root = '/big_fast_drive/orm/exsample_data/'

dsm = [  DatasetMetadata(name='dashcam_unified_ng',
                    boxpath=f'{root}/dashcam_unified_ng.track_ids.clean.parquet',
                    track_id='tid',
                    frame_id='frame_idx',
                    video_len=1164226,
                    category='category',
                    categories=[
                        'person',
                        'bicycle',
                        'bus',
                        'truck',
                        'traffic light',
                        'fire hydrant',
                        'stop sign'],
                    default_chunks=(f'{root}/dash_metadata_by_frame.parquet', 'chunk_id'),
                         # this chunk id already is split by file and then 30 minutes
                    logitpath=f'{root}/dashcam_logits_secon_version.parquet',
                    logitsplit='subset'),
    DatasetMetadata(name='bdd1k',
                    boxpath=f'{root}/bdd1k.track_ids.clean.parquet',
                    track_id='tid',
                    frame_id='frame_idx',
                    video_len=389662,  # 1/3 frames of 1k ~40 second clips.
                    category='category',
                    categories=['person',
                                'rider',
                                'bus',
                                'truck',
                                'bike',
                                'motor',
                                'traffic light',
                                'traffic sign',],
                    default_chunks=(f'{root}/bdd_metadata_by_frame.parquet', 'clip_id'),
                    logitpath=f'{root}/bdd_logits_v0.parquet',
                    logitsplit='subset'),
        DatasetMetadata(name='bdd_mot_labels',
                boxpath=f'{root}/bdd_mot_labels.parquet',
                track_id='id',
                frame_id='frame_idx',
                video_len=None,
                category='category',
                categories=['car','pedestrian','truck','bicycle','rider','bus','motorcycle'],
                default_chunks=(f'{root}/bdd_mot_frames.parquet','video_name'),
                logitpath=None,
                logitsplit=None),
    DatasetMetadata(name='jackson-town-square',
                        boxpath=f'{root}/jackson-town-square_boxes_tids_filtered.parquet',
                        track_id='tid', frame_id='frame_idx', category='category',
                        categories=['car', 'person', 'truck', 'bus', 'dog', 'motorcycle'],
                        video_len=None, default_chunks=None,
                        logitpath=None, logitsplit=None),
   DatasetMetadata(name='archie-day',
                    boxpath=f'{root}/archie-day_boxes_tids_filtered.parquet',
                    track_id='tid', frame_id='frame_idx', category='category',
                    categories=['car', 'bicycle', 'person', 'motorcycle', 'truck', 'bus'],
                    video_len=None, default_chunks=None,
                    logitpath=None, logitsplit=None),
   DatasetMetadata(name='amsterdam',
                    boxpath=f'{root}/amsterdam_boxes_tids_filtered.parquet',
                    track_id='tid', frame_id='frame_idx', category='category',
                    categories=['boat', 'car', 'person', 'bicycle', 'truck', 'motorcycle', 'dog'],                        video_len=None, default_chunks=None,
                    logitpath=None, logitsplit=None)
]


bms = []
for m in dsm:
    bms.append(BenchmarkDataset(m))


def generate_configs(num_reps=11):
    batch_size=10
    combos = []
    for ds in bms:
        for _ in range(num_reps):
            baselines = named_prod(
                order=['random', 'random+'],
                split=['nosplit'],
                batch_size=[batch_size],
                object_class=ds.categories,
                dataset=[ds],
                score_method=[None]
            )
            exsample = named_prod(
                order=['random+'],
                split=['default'],
                batch_size=[batch_size],
                object_class=ds.categories,
                dataset=[ds],
                score_method=[(DATA_ROOT, dict(stat='gt', exp_model='ts'))]
            )

            combos.append(baselines)
            combos.append(exsample)

    configs = []
    for c in combos:
        for tup in c:
            configs.append(tup)
    return configs

import random
from ray.util.multiprocessing import Pool
print('creating pool with {} cpus'.format(args.num_cpus))
pool = Pool(args.num_cpus)

cfgs = generate_configs(args.reps)
threshold = args.threshold

def run_config(config):
    exp = make_experiment(config)
    exp.run_up_to(threshold=threshold, disable_tqdm=True)
    df = exp.get_results()
    return df

print('running pool.map...')
results = pool.map(run_config, cfgs)
res_new = pd.concat(results, ignore_index=True)
fname = 'sample_results_{}.parquet'.format(random.randint(0,2**20))
res_new.to_parquet(fname)
print(fname)