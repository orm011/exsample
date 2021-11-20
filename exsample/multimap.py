import numpy as np

class Sparse2DenseMapper(object):
    def __init__(self, id_bound):
        self.next_id = 0
        self.id_map = np.zeros(id_bound + 1, dtype=np.int32) - 1  # -1 means no id

    def __getitem__(self, sparse_id):
        lk = self.id_map[sparse_id]  # must be smaller than id_bound...
        if lk >= 0:
            return lk

        dense_id = self.next_id
        self.id_map[sparse_id] = dense_id
        self.next_id += 1
        return dense_id

def testDenseIdMapper():
    max_id = 2 ** 30
    testids = np.random.randint(0, high=max_id, size=20)
    testids = list(set(testids))
    idm = Sparse2DenseMapper(id_bound=max_id)

    for (i, tid) in enumerate(testids):
        mid = idm[tid]
        assert mid == i

        mid2 = idm[tid]
        assert mid2 == mid

    for (i, tid) in enumerate(testids):
        mid = idm[tid]
        assert mid == i

class ArrayMultiMap(object):
    def __init__(self, keys : np.ndarray, values : np.ndarray):
        order = np.argsort(keys)
        self.okeys = keys[order]
        self.ovalues = values[order]

        keyval, index, counts = np.unique(self.okeys, return_index=True, return_counts=True)
        assert self.okeys.min() >= 0
        idxlen = self.okeys.max() + 1
        poslen = np.zeros((idxlen, 2), dtype=np.int)
        poslen[keyval, 0] = index
        poslen[keyval, 1] = counts
        self.poslen = poslen

    def __getitem__(self, k):
        if k >= self.poslen.shape[0]:  # empty result
            return self.ovalues[0:0]

        (vstart, vlen) = self.poslen[k]
        values = self.ovalues[vstart:vstart + vlen]
        return values


def test_multimap():
    from collections import defaultdict, OrderedDict

    test_dict = defaultdict(lambda: [], OrderedDict([(7, [20]), (1, [10, 20]), (3, [10]), (5, [11, 12])]))

    ks = []
    vs = []
    for (k, vr) in test_dict.items():
        vs.extend(vr)
        ks.extend([k for _ in vr])

    amm = ArrayMultiMap(keys=np.array(ks), values=np.array(vs))

    for k in range(100):
        assert set(test_dict[k]) == set(amm[k])