
import new_coviar
import pandas as pd
import numpy as np
import os
import numpy as np
import time
import fire


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



# os.path.exists(path)

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


def get_split_meta(cv, part_id, num_parts):
    assert part_id < num_parts
    split_size = int(np.ceil(cv.metadata.index.shape[0] / num_parts))
    assignments = cv.metadata.index // split_size
    return cv.metadata.iloc[assignments == part_id]

ten_min = ((10 * 60 * 30))

def process_range(path, part_no, total_parts, dh=100, dw=150):
    cv = CoviarVideo(path)

    sum_mv = []
    sum_rs = []
    sum_mv_boundary = []
    sum_rs_boundary = []
    frame_ids = []

    meta = get_split_meta(cv, part_no, total_parts)

    start = time.time()
    for (i, tup) in enumerate(meta.itertuples()):
        offset_hint = cv.gop_offsets.iloc[tup.gop_number]
        (mv, rs) = new_coviar.load(path, tup.gop_number, tup.frame_pos, 2, 0, offset_hint)
        mv = np.square(mv)
        rs = np.square(rs)

        smv = mv.reshape(-1).sum()
        srs = rs.reshape(-1).sum()

        sum_mv.append(smv)
        sum_rs.append(srs)

        mv[dh:-dh, dw:-dw, :] = 0
        rs[dh:-dh, dw:-dw, :] = 0

        smv2 = mv.reshape(-1).sum()
        srs2 = rs.reshape(-1).sum()
        sum_mv_boundary.append(smv2)
        sum_rs_boundary.append(srs2)

        frame_ids.append(tup.Index)

        if i % ten_min == 0:
            print(i, time.time() - start)

    df = pd.DataFrame({'mv': sum_mv, 'rs': sum_rs, 'mv_boundary': sum_mv_boundary,
                       'rs_boundary': sum_rs_boundary}, index=frame_ids)

    opath = (path.replace('.mp4', '') + '.{}_of_{}.opt.parquet').format(part_no, total_parts)
    print(opath)
    df.to_parquet(opath)
    return 0


if __name__ == '__main__':
    fire.Fire(process_range)