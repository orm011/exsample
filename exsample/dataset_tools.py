import torch
import torch.utils.data
import PIL
import os
import torchvision.transforms as T

std_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

import skvideo.io

class SkVideoClip(torch.utils.data.IterableDataset):
    def __init__(self, path, tx=None):
        self.path = path
        self.tx = (lambda x: x) if tx is None else tx

    def __iter__(self):
        vr = skvideo.io.vreader(self.path)
        for frame in vr:
            yield self.tx(frame)

class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, path, extension='.png', tx=None):
        self.path = path
        self.extension = extension
        self.files = sorted([f for f in os.listdir(path) if f.endswith(extension)])
        self.tx = (lambda x: x) if (tx is None) else tx

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.tx(PIL.Image.open(os.path.join(self.path, self.files[idx])))

class HCatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, xforms=lambda x: x):
        self.datasets = tuple(datasets)
        self.__len = min([len(x) for x in self.datasets])
        self.xforms = xforms

    def __len__(self):
        return self.__len

    def __getitem__(self, idx):
        ans = [ds[idx] for ds in self.datasets]
        return self.xforms(ans)

class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df, index_var, max_idx=None, xforms=None):
        self.df = df
        self.xforms = (lambda x: x) if xforms is None else xforms
        self.max_idx = (df[index_var].max()) if (max_idx is None) else max_idx
        self.index_var = index_var

    def __len__(self):
        return self.max_idx + 1

    def __getitem__(self, idx):
        assert idx <= self.max_idx
        quals = self.df[self.index_var] == idx
        return self.xforms(self.df[quals])

class ZarDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        import zarr

        self.path = path
        self.arr = zarr.open(path, 'r')
        self.shape = self.arr.shape

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return self.arr[idx]