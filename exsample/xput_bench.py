import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
device = torch.device('cuda:1')
#trn10 = trn.create_trn('trn10', 1, weights='random').eval().to(device)
model_101 = models.resnet101().eval().to(device)
model_18 = models.resnet18().eval().to(device)

def make_ts(resize):
    ts = transforms.Compose([
        #                   transforms.Lambda(lambda x : x[...,[2,1,0]]), # bgr 2 rgb
        transforms.ToPILImage(),
        transforms.Resize(resize) if resize is not None else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return ts

def preload_dataset(dataset, preload_len=1000, **kwargs):
    data = []
    bs = 20
    for i in range(0, preload_len, bs):
        b = dataset[i:i+bs]
        data.append(b)

    data = torch.cat(data)
    assert data.shape[0] == preload_len
    return data


def bench(dataset, model, batch_size=10, nframes=100, shuffle=False, **kwargs):
    if shuffle:
        assert len(dataset) > nframes  # warning: random access over small data
        idx = np.random.permutation(len(dataset))[:nframes]
    else:
        reps = nframes // len(dataset) + 1 * (nframes % len(dataset) > 0)
        idx = np.arange(len(dataset))[:nframes]
        idx = np.concatenate([idx] * reps)

    start = time.time()
    rets = []
    tot = 0
    if shuffle:
        batch_size = 20
    else:
        batch_size = 20

    def samp():
        for i in range(0, len(idx), batch_size):
            indices = idx[i:i + batch_size]
            yield indices

    def frame_iter():
        if not torch.is_tensor(dataset):
            dataset._dec = None  # force re-init
            if shuffle:
                nw = 3
            else:
                nw = 4
            dl = DataLoader(dataset, batch_sampler=samp(),
                            num_workers=nw, pin_memory=True)

            for (i, b) in enumerate(dl):
                yield b
                if (i + 1) * batch_size >= nframes:
                    break
        else:
            for i in range(0, len(idx), batch_size):
                image_tensors = dataset[idx[i:i + batch_size]]
                yield image_tensors

    fiter = frame_iter()
    for image_tensors in fiter:
        #     for i in range(0, len(idx), batch_size):
        #         image_tensors = dataset[idx[i:i+batch_size]]
        assert torch.is_tensor(image_tensors)
        assert image_tensors.shape[0] == batch_size
        if model is not None:
            with torch.no_grad():
                ret = model(image_tensors.to(device)).to('cpu')
            rets.append(ret)
        else:
            rets.append(image_tensors)

    end = time.time()
    rt = torch.cat(rets)
    assert rt.shape[0] >= nframes
    xput = rt.shape[0] / (end - start)
    print('{:.0f} fps'.format(xput))
    return {'nframes': rt.shape[0],
            'duration': end - start,
            'xput': xput}


def stage_timing1(ds, ds_pr, ds_small, ds_pr_small,
                  exp_model, cheap_model, small_n=100, large_n=200, batch_size=4, num_workers=0):
    res = []

    res.append(bench(ds, model=exp_model, nframes=small_n, shuffle=False,
                     num_workers=num_workers,
                     batch_size=batch_size))
    res[-1]['phase'] = 'label'

    res.append(bench(ds_pr_small, model=cheap_model.train(), nframes=large_n, shuffle=False,
                     num_workers=0, batch_size=batch_size))
    res[-1]['phase'] = 'train'

    res.append(bench(ds_small, model=cheap_model.eval(), nframes=large_n, shuffle=False,
                     num_workers=num_workers,
                     batch_size=batch_size))
    res[-1]['phase'] = 'score'

    res.append(bench(ds, model=exp_model, nframes=small_n, shuffle=True,
                     num_workers=num_workers,
                     batch_size=batch_size))
    res[-1]['phase'] = 'sample'
    return pd.DataFrame.from_records(res)


def stage_timing2(ds, ds_pr, ds_small, ds_pr_small,
                  exp_model, cheap_model, small_n=100, large_n=200, batch_size=4, num_workers=0):
    res2 = []

    res2.append(bench(ds_pr, model=exp_model, nframes=small_n, shuffle=False,
                      num_workers=0, batch_size=batch_size))
    res2[-1]['phase'] = 'exp_model_only'

    res2.append(bench(ds_pr_small, model=cheap_model, nframes=large_n, shuffle=False,
                      num_workers=0, batch_size=batch_size))
    res2[-1]['phase'] = 'cheap_model_only'

    res2.append(bench(ds, model=None, nframes=small_n, shuffle=True,
                      num_workers=num_workers, batch_size=batch_size))
    res2[-1]['phase'] = 'random_io_only'

    res2.append(bench(ds_small, model=None, nframes=large_n, shuffle=False,
                      num_workers=num_workers, batch_size=batch_size))
    res2[-1]['phase'] = 'seq_io_only'

    return pd.DataFrame.from_records(res2)


def get_xput(a):
    return a['nframes'] / a['duration']


def post_process(ds, ds_name, a):
    a = a.assign(dataset=ds_name)
    a = a.assign(xput=get_xput(a))
    n_epochs = 1
    nframes = ds.video.len
    trlen = 120000
    nfrms = defaultdict(lambda: np.nan, **{'label': trlen,
                                           'train': n_epochs * trlen,
                                           'score': nframes - trlen,
                                           'sample': .1 * (nframes - trlen)})

    a = a.assign(total_frames=a.phase.map(lambda x: nfrms[x]))
    a = a.assign(est_seconds=(a.total_frames / a.xput).astype('int'))
    return a

sqv = stage_timing2(*square_dss,exp_model=model_101,cheap_model=model_18,
             small_n=500, large_n=2000, batch_size=10, num_workers=0)
dasha = stage_timing1(*dash_dss, exp_model=model_101,cheap_model=model_18,
             small_n=300, large_n=1000, batch_size=10, num_workers=0)
dashv = stage_timing2(*dash_dss,exp_model=model_101,cheap_model=model_18,  small_n=300, large_n=1000, batch_size=10, num_workers=0)
sqa = stage_timing1(*square_dss, exp_model=model_101,cheap_model=model_18,
             small_n=500, large_n=2000, batch_size=10, num_workers=0)