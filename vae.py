import sys
from collections import defaultdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import torchvision
from torch.optim import Adam
from itertools import chain, count
import torch
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F

# TODO: VQ VAE may be worth doing. but maybe as a separate repo.
from nets import E1, D1
import argparse
import utils

H = utils.AttrDict()
H.bs = 512
H.z_size = 128
H.bn = 0
H.device = 'cuda'
H.log_n = 1000
H.done_n = int(1e5)
H.b = 1.0
H.name = './logs/'
H.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called

def dump_logger(logger, writer, i, H):
    print('='*30)
    print(i)
    for key in logger:
        val = np.mean(logger[key])
        writer.add_scalar(key, val, i)
        print(key, val)
    print(H.full_cmd)
    print('='*30)
    return defaultdict(lambda: [])

def plot_samples(writer, *args):
    l = []
    for a in args:
        upr = utils.unproc(a)
        l += [upr, np.zeros_like(upr)]
    img = np.concatenate(l, axis=1)
    plt.imsave('test.png', img)
    writer.add_image('test', img/255.0, i, dataformats='HWC')

if __name__ == '__main__':
    from utils import CIFAR, MNIST
    import argparse
    parser = argparse.ArgumentParser()
    for key, value in H.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    H = parser.parse_args()

    writer = SummaryWriter(H.name)
    logger = dump_logger({}, writer, 0)

    ds = CIFAR(H)
    #ds = MNIST(device)
    encoder = E1(H).to(H.device)
    decoder = D1(H).to(H.device)
    optimizer = Adam(chain(encoder.parameters(), decoder.parameters()), lr=1e-4)

    for i in count():
        optimizer.zero_grad()
        batch = ds.sample_batch(H.bs)
        prior_loss, code, mu = encoder(batch)
        recondist = decoder(code)

        recon_loss = -recondist.log_prob(batch)
        loss = (H.b*prior_loss + recon_loss.mean((-1, -2, -3))).mean()
        loss.backward()
        optimizer.step()
        logger['total_loss'] += [loss.detach().cpu()]
        logger['prior_loss'] += [prior_loss.mean().detach().cpu()]
        logger['recon_loss'] += [recon_loss.mean().detach().cpu()]

        if i % H.log_n == 0:
            encoder.eval()
            decoder.eval()
            logger = dump_logger(logger, writer, i, H)
            reconmu = decoder(mu[:10]).mean
            reconsamp = decoder(torch.randn(mu[:10].shape).to(H.device)).mean
            plot_samples(writer, batch[:10], reconmu, reconsamp)
            writer.flush()
            encoder.train()
            decoder.train()
        if i >= H.done_n:
            break
