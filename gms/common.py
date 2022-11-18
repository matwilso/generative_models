import importlib
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import yaml
from einops import rearrange, repeat
from scipy.linalg import fractional_matrix_power
from torch import distributions as tdib
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import MNIST

# BASIC DEFS AND UTILS


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def prefix_dict(name, dict):
    return {name + key: dict[key] for key in dict}


def convert_camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def discover_models():
    """
    discover all the models that subclass GM
    """
    models = {}
    path = Path(__file__).parent
    for file in path.rglob('*.py'):
        if '__init__' in file.name or 'main' in file.name:
            continue
        relative_path = file.relative_to(path.parent)
        import_format = str(relative_path).replace('/', '.').replace('.py', '')
        module = importlib.import_module(import_format)
        for key in dir(module):
            obj = getattr(module, key)
            # check if the class is a subclass of GM
            if type(obj) == type and issubclass(obj, GM):
                models[convert_camel_to_snake(key)] = obj
    return models


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x


def dump_logger(logger, writer, i, G):
    print('=' * 30)
    print(i)
    for key in logger:
        val = np.mean(logger[key])
        writer.add_scalar(key, val, i)
        print(key, val)

    G.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
    G.commit_hash = (
        subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    )
    print(G.full_cmd)
    with open(Path(G.logdir) / 'hps.yaml', 'w') as f:
        yaml.dump(dict(G), f, width=float("inf"))
    print('=' * 30)
    writer.flush()
    return defaultdict(lambda: [])


def args_type(default):
    if isinstance(default, bool):
        return lambda x: bool(['False', 'True'].index(x))
    if isinstance(default, int):
        return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, Path):
        return lambda x: Path(x).expanduser()
    return type(default)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


# DATA


def load_mnist(bs, binarize=True, pad32=False):

    tfs = [transforms.ToTensor()]
    if binarize:
        tfs += [lambda x: (x > 0.5).float()]
    else:
        tfs += [lambda x: x.float()]
        tfs += [lambda x: 2 * x - 1]
    if pad32:
        tfs += [lambda x: F.pad(x, (2, 2, 2, 2))]
    transform = transforms.Compose(tfs)
    train_dset = MNIST('data', transform=transform, train=True, download=True)
    test_dset = MNIST('data', transform=transform, train=False, download=True)

    train_loader = data.DataLoader(
        train_dset,
        batch_size=bs,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
    )
    test_loader = data.DataLoader(
        test_dset,
        batch_size=bs,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
    )
    return train_loader, test_loader


# MODEL DEFS AND UTILS


class GM(nn.Module):
    """
    GenerativeModel
    """

    DG = (
        AttrDict()
    )  # default configuration. can be customized across models, both by changing the default value and by adding new options

    def __init__(self, G):
        super().__init__()
        self.G = G
        ## make the name be the name of the file (without .py)
        # self.name = Path(inspect.getfile(self.__class__)).with_suffix('').name
        self.optimizer = None

    def save(self, path, test_x=None, test_y=None):
        model_path = path / f'model.pt'
        torch.save(self.state_dict(), model_path)

    def train_step(self, x, y):
        """take one step on a batch to update the network"""
        assert hasattr(
            self, 'loss'
        ), 'you are using the default train_step. this requires you to define a loss function that returns loss, metrics'
        if self.optimizer is None:
            self.optimizer = Adam(self.parameters(), self.G.lr)
        self.optimizer.zero_grad()
        loss, metrics = self.loss(x, y)
        loss.backward()
        self.optimizer.step()
        return metrics

    def evaluate(self, writer, x, y, epoch):
        assert (
            False
        ), "you need to implement the evaluate method. make some samples or something."

    def end_epoch(self):
        pass


def write_grid(writer, tag, x, epoch):
    assert tuple(x.shape) == (25, 1, 28, 28)
    x = rearrange(x, '(n1 n2) c h w -> c (n1 h) (n2 w)', n1=5, n2=5)
    writer.add_image(tag, x, epoch)


def write_gridvid(writer, tag, x, epoch):
    T = x.shape[0]
    assert tuple(x.shape[1:]) == (25, 1, 28, 28)
    vid = rearrange(x, 't (n1 n2) c h w -> t c (n1 h) (n2 w)', n1=5, n2=5)[None]
    vid = repeat(vid, 'b t c h w -> b t (repeat c) h w', repeat=3)
    writer.add_video(
        tag,
        vid,
        epoch,
        fps=min(T // 3, 60),
    )


class Autoreg(GM):
    def evaluate(self, writer, x, y, epoch):
        samples, gen = self.sample(25)
        # convert to a 5x5 grid for sample
        write_grid(writer, 'samples', samples, epoch)
        write_gridvid(writer, 'sampling_process', gen, epoch)


class Arbiter(GM):
    def save(self, path, test_x, test_y=None):
        model_path = path / 'model.jit.pt'
        jit_enc = torch.jit.trace(self, test_x)
        torch.jit.save(jit_enc, model_path)


class CategoricalHead(nn.Module):
    """take logits and produce a multinomial distribution independently"""

    def __init__(self, in_n, out_n, G):
        super().__init__()
        self.layer = nn.Linear(in_n, out_n)

    def forward(self, x):
        x = self.layer(x)
        return tdib.Multinomial(logits=x)


class BinaryHead(nn.Module):
    """take logits and produce a bernoulli distribution independently"""

    def __init__(self, in_n, out_n, G):
        super().__init__()
        self.layer = nn.Linear(in_n, out_n)

    def forward(self, x):
        x = self.layer(x)
        return tdib.Bernoulli(logits=x)


def append_location(x):
    """add xy coords to every pixel"""
    XY = torch.stack(
        torch.meshgrid(torch.linspace(0, 1, 28), torch.linspace(0, 1, 28)), 0
    ).to(x.device)
    return torch.cat([x, XY[None].repeat_interleave(x.shape[0], 0)], 1)


# EVAL AND METRIC UTILS


def combine_imgs(arr, row=5, col=5):
    """takes batch of video or image and pushes the batch dim into certain image shapes given by b,row,col"""
    if len(arr.shape) == 4:  # image
        BS, _, H, W = arr.shape
        assert BS == row * col and H == W == 28, (BS, row, col, H, W)
        x = arr.reshape([row, col, 28, 28]).permute(0, 2, 1, 3).flatten(0, 1).flatten(-2)
        return x
    elif len(arr.shape) == 5:  # video
        BS, T, _, H, W = arr.shape
        assert BS == row * col and H == W == 28, (BS, T, row, col, H, W)
        x = (
            arr.reshape([row, col, T, 28, 28])
            .permute(2, 0, 3, 1, 4)
            .flatten(1, 2)
            .flatten(-2)
        )
        return x
    else:
        raise NotImplementedError()


def compute_fid(x, y):
    """
    FID / Wasserstein Computation
    https://en.wikipedia.org/wiki/Wasserstein_metric#Normal_distributions
    https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance
    """
    try:
        assert x.ndim == 2 and y.ndim == 2
        # aggregate stats from this batch
        pmu = np.mean(x, 0)
        pcov = np.cov(x, rowvar=False)
        tmu = np.mean(y, 0)
        tcov = np.cov(y, rowvar=False)
        assert pcov.shape[0] == x.shape[-1]
        # compute FID equation
        fid = np.mean((pmu - tmu) ** 2) + np.trace(
            pcov + tcov - 2 * fractional_matrix_power(pcov.dot(tcov), 0.5)
        )
        # TODO: this is somewhat concerning i got an imaginary number before.
        return fid.real
    except Exception:
        return np.nan


def precision_recall_f1(*, real, gen, k=3):
    """
    precision = realistic. fraction of generated images that are realistic
    recall = coverage. fraction of data manifold covered by generator

    precision = how often do you generate something that is closer to a real sample than other real samples.
    recall = how often is there a real sample that is closer to your generated one than other generated ones are.

    k determines the strictness of the manifold check. if k=3, then a point from set_b needs
    to be closer to a point in set_a than 3 other points in set_a.

    real: (NxZ)
    gen: (NxZ)
    """
    # TODO: is there a way we can do this online? like basically compute distance matrixes for very large datasets. then we just need 3 distance matrixes.

    def _manifold_estimate(set_a, set_b, k=3):
        """https://arxiv.org/abs/1904.06991"""
        # compute manifold
        d = torch.cdist(set_a, set_a)
        radii = torch.topk(d, k + 1, largest=False).values[..., -1:]
        # eval
        d2 = torch.cdist(set_a, set_b)
        return (d2 < radii).any(0).float().mean()

    precision = _manifold_estimate(real, gen, k)
    recall = _manifold_estimate(gen, real, k)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {'precision': precision, 'recall': recall, 'f1': f1}
