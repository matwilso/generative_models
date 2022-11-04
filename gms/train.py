import argparse
import os
import sys
import time
from itertools import count
from pathlib import Path

import torch
import yaml
from ignite.metrics import FID
from torch.utils.tensorboard import SummaryWriter

from gms import common

# TRAINING SCRIPT

# cmd line args with their default values. models can add additional args of their own
# DG = (D)efault G, where G is a config object. Single letters C, H, F were already taken, so G.
DG = common.AttrDict()
DG.model = 'vae'
DG.bs = 64
DG.hidden_size = 256
DG.device = 'cuda'
DG.epochs = 50
DG.save_n = 1
DG.logdir = Path('./logs/')
DG.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
DG.lr = 3e-4
DG.class_cond = 0
DG.binarize = 1
DG.pad32 = 0
DG.mode = 'train'
DG.weights_from = Path('.')
DG.arbiter_dir = Path('.')
DG.debug_eval = 0


def load_model_and_data():
    """
    Parse command line arguments, load model and data.
    """
    # initial parse of the args to allow user to specify model, weights_from, etc
    parser = argparse.ArgumentParser()
    for key, value in DG.items():
        parser.add_argument(f'--{key}', type=common.args_type(value), default=value)
    tempG, _ = parser.parse_known_args()

    # grab the extra parameters and default argumments from the model or weights_from hps.yaml
    defaults = {}
    if tempG.weights_from != Path('.'):
        # if we have a weights_from, load the config from there.
        loaded_hp_file = tempG.weights_from.parent / 'hps.yaml'
        with open(loaded_hp_file) as f:
            loadedG = common.AttrDict(yaml.load(f, Loader=yaml.Loader).__dict__)
        for key, value in loadedG.items():
            defaults[key] = value
            if key not in tempG:
                parser.add_argument(f'--{key}', type=type(value), default=value)
        Model = common.discover_models()[loadedG.model]
    else:
        Model = common.discover_models()[tempG.model]
        for key, value in Model.DG.items():
            defaults[key] = value
            if key not in tempG:
                parser.add_argument(f'--{key}', type=type(value), default=value)
        defaults['logdir'] = tempG.logdir / tempG.model

    # do the final parse of cmd line args for what user passes in and instantiate everything
    parser.set_defaults(**defaults)
    G = common.AttrDict(parser.parse_args().__dict__)
    model = Model(G=G).to(G.device)
    if G.weights_from != Path('.'):
        model.load_state_dict(torch.load(G.weights_from, map_location=G.device))
    train_ds, test_ds = common.load_mnist(G.bs, binarize=G.binarize, pad32=G.pad32)
    print('num_vars', common.count_vars(model))

    return model, train_ds, test_ds, G


def train(model, train_ds, test_ds, G):
    writer = SummaryWriter(G.logdir)
    logger = common.dump_logger({}, writer, 0, G)

    # TRAINING LOOP
    for epoch in count(1):
        # TRAIN
        train_time = time.time()
        for batch in train_ds:
            batch[0], batch[1] = batch[0].to(G.device), batch[1].to(G.device)
            # TODO: see if we can just use loss and write the gan such that it works.
            metrics = model.train_step(batch[0])
            for key in metrics:
                logger[f'{G.model}/train/{key}'] += [metrics[key].detach().cpu()]
            if G.debug_eval:
                break

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
                        logger[f'{G.model}/test/{key}'] += [
                            test_metrics[key].detach().cpu()
                        ]
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
        logger['num_vars'] = common.count_vars(model)
        logger = common.dump_logger(logger, writer, epoch, G)
        if epoch % G.save_n == 0:
            path = G.logdir / f'model_{epoch}.pt'
            latest_path = G.logdir / 'model.pt'
            print("SAVED MODEL", path)
            torch.save(model.state_dict(), path)
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(path)
            if G.model == 'arbiter':
                model.save(G.logdir, batch[0])
        if epoch >= G.epochs:
            break


if __name__ == '__main__':
    model, train_ds, test_ds, G = load_model_and_data()
    train(model, train_ds, test_ds, G)
