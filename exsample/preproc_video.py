import h5py
import os

def video2hd5(vid, outputfile, limit=None, batch_size=1000):
    if limit is None:
        limit = len(vid)

    basic_shape = vid[0].shape
    dtp = vid[0].numpy().dtype

    drnm = os.path.dirname(outputfile)
    os.makedirs(drnm, exist_ok=True)

    with h5py.File(outputfile, 'w') as f:
        dset = f.create_dataset("frames", dtype=dtp,
                                shape=(0,) + basic_shape,
                                chunks=(batch_size,) + basic_shape,
                                maxshape=(None,) + basic_shape)

        for i in range(0, limit, batch_size):
            ends = min(i + batch_size, limit)
            sz = ends - i
            dset.resize((dset.shape[0] + sz,) + basic_shape)
            dset[-sz:, ...] = vid[i:i + sz]