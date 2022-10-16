import argparse
import pathlib
import sys
import time
from itertools import count

import torch
from torch.utils.tensorboard import SummaryWriter

from gms import autoregs, common, diffusion, gans, vaes

# TRAINING SCRIPT

G = common.AttrDict()
G.model = 'vae'
G.bs = 64
G.hidden_size = 256
G.device = 'cuda'
G.num_epochs = 50
G.save_n = 100
G.logdir = pathlib.Path('./logs/')
G.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
G.lr = 3e-4
G.class_cond = 0
G.binarize = 1
G.pad32 = 0

if __name__ == '__main__':
    # PARSE CMD LINE
    parser = argparse.ArgumentParser()
    for key, value in G.items():
        parser.add_argument(f'--{key}', type=common.args_type(value), default=value)
    tempG, _ = parser.parse_known_args()
    # SETUP
    Model = {
        'rnn': autoregs.RNN,
        'made': autoregs.MADE,
        'wavenet': autoregs.Wavenet,
        'pixelcnn': autoregs.PixelCNN,
        'gatedcnn': autoregs.GatedPixelCNN,
        'transformer': autoregs.TransformerCNN,
        'vae': vaes.VAE,
        'vqvae': vaes.VQVAE,
        'gan': gans.GAN,
        'diffusion': diffusion.DiffusionModel,
    }[tempG.model]
    defaults = {'logdir': tempG.logdir / tempG.model}
    for key, value in Model.DG.items():
        defaults[key] = value
        if key not in tempG:
            parser.add_argument(f'--{key}', type=type(value), default=value)
    parser.set_defaults(**defaults)
    G = parser.parse_args()
    model = Model(G=G).to(G.device)
    writer = SummaryWriter(G.logdir)
    logger = common.dump_logger({}, writer, 0, G)
    train_ds, test_ds = common.load_mnist(G.bs, binarize=G.binarize, pad32=G.pad32)
    num_vars = common.count_vars(model)
    print('num_vars', num_vars)

    # TRAINING LOOP
    for epoch in count():
        # TRAIN
        train_time = time.time()
        for batch in train_ds:
            batch[0], batch[1] = batch[0].to(G.device), batch[1].to(G.device)
            # TODO: see if we can just use loss and write the gan such that it works.
            metrics = model.train_step(batch[0])
            for key in metrics:
                logger[key] += [metrics[key].detach().cpu()]
        logger['dt/train'] = time.time() - train_time
        # TEST
        model.eval()
        with torch.no_grad():
            # if we define an explicit loss function, use it to test how we do on the test set.
            if hasattr(model, 'loss'):
                for test_batch in test_ds:
                    test_batch[0], test_batch[1] = test_batch[0].to(
                        G.device
                    ), test_batch[1].to(G.device)
                    test_loss, test_metrics = model.loss(test_batch[0])
                    for key in test_metrics:
                        logger['test/' + key] += [test_metrics[key].detach().cpu()]
            else:
                test_batch = next(iter(test_ds))
                test_batch[0], test_batch[1] = test_batch[0].to(G.device), test_batch[
                    1
                ].to(G.device)
            # run the model specific evaluate function. usually draws samples and creates other relevant visualizations.
            eval_time = time.time()
            model.evaluate(writer, test_batch[0], epoch)
            logger['dt/evaluate'] = time.time() - eval_time
        model.train()
        # LOGGING
        logger['num_vars'] = num_vars
        logger = common.dump_logger(logger, writer, epoch, G)
        if epoch % G.save_n == 0:
            path = G.logdir / 'model.pt'
            print("SAVED MODEL", path)
            torch.save(model.state_dict(), path)
        if epoch >= G.num_epochs:
            break
