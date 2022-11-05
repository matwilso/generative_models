import argparse
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
# DG = (D)efault G, where G is a config object. Single letters C, H, F were already taken, so G. lol
DG = common.AttrDict()
DG.model = 'vae'
DG.bs = 64
DG.hidden_size = 256
DG.device = 'cuda'
DG.epochs = 50
DG.save_n = 5
DG.logdir = Path('./logs/')
DG.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
DG.lr = 3e-4
DG.class_cond = 0
DG.binarize = 1
DG.pad32 = 0
DG.mode = 'train'
DG.weights_from = Path('.')  # path to a model that you want to load and keep training
DG.arbiter = Path('.')  # for computing a latent space for evaluation metrics
DG.classifier = Path('.')  # for computing a latent space for evaluation metrics
DG.eval_heavy = 1
DG.skip_training = 0


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

    # set the arbiter dir to the default model provided by this repo
    is_binarized = defaults['binarize'] if 'binarize' in defaults else tempG.binarize
    defaults['arbiter'] = (
        Path('weights/encoder_binary.pt')
        if is_binarized
        else Path('weights/encoder_continuous.pt')
    )

    # do the final parse of cmd line args for what user passes in and instantiate everything
    parser.set_defaults(**defaults)
    G = common.AttrDict(parser.parse_args().__dict__)
    model = Model(G=G).to(G.device)
    if G.weights_from != Path('.'):
        model.load_state_dict(torch.load(G.weights_from, map_location=G.device))
    train_ds, test_ds = common.load_mnist(G.bs, binarize=G.binarize, pad32=G.pad32)
    print('num_vars', common.count_vars(model))
    arbiter = torch.jit.load(G.arbiter).to(G.device) if G.eval_heavy else None

    return model, train_ds, test_ds, arbiter, G


@torch.inference_mode()
def eval_heavy(logger, model, test_ds, arbiter, G):
    """
    More computationally intensive eval that draws many samples from the generative model and computes
    FID and precision/recall metrics.
    """
    TOTAL_SAMPLES = 500  # beyond about 500, we started getting diminishing returns

    model.eval()

    # sample the model and get latent vectors for samples as well as test set examples
    # so that we can then compare the two distributions (sampled vs. test).
    sample_ct = 0
    all_z_sample = []
    all_z_real = []
    fid_buffer = FID(num_features=64, feature_extractor=arbiter, device=G.device)
    for test_batch in test_ds:
        test_batch[0], test_batch[1] = test_batch[0].to(G.device), test_batch[1].to(
            G.device
        )
        bs = test_batch[0].shape[0]
        samp = model.sample(bs)
        fid_buffer.update((samp, test_batch[0]))
        z_samp = arbiter(samp)
        z_real = arbiter(test_batch[0])
        all_z_real.append(z_real)
        all_z_sample.append(z_samp)
        sample_ct += bs
        if sample_ct >= TOTAL_SAMPLES:
            break

    ingite_fid = fid_buffer.compute()
    z_samp = z_samp
    z_real = z_real
    fid = common.compute_fid(z_samp.cpu().numpy(), z_real.cpu().numpy())
    precision, recall, f1 = common.precision_recall_f1(z_real, z_samp)
    model.train()
    metrics = {
        'ingite_fid': ingite_fid,
        'fid': fid,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    for key, val in metrics.items():
        logger[f'eval/{key}'] += [common.to_numpy(val)]


def train(model, train_ds, test_ds, arbiter, G):
    writer = SummaryWriter(G.logdir)
    logger = common.dump_logger({}, writer, 0, G)

    # TRAINING LOOP
    for epoch in count(1):
        # TRAIN
        train_time = time.time()
        for batch in train_ds:
            if G.skip_training:
                break
            batch[0], batch[1] = batch[0].to(G.device), batch[1].to(G.device)
            # TODO: see if we can just use loss and write the gan such that it works.
            metrics = model.train_step(batch[0], batch[1])
            for key in metrics:
                logger[f'{G.model}/train/{key}'] += [metrics[key].detach().cpu()]

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
                    test_loss, test_metrics = model.loss(test_batch[0], test_batch[1])
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
            model.evaluate(writer, test_batch[0], test_batch[1], epoch)
            logger['dt/eval'] = time.time() - eval_time
        model.train()
        # LOGGING
        logger['num_vars'] = common.count_vars(model)
        if epoch % G.save_n == 0:
            path = G.logdir / f'model_{epoch}.pt'
            latest_path = G.logdir / 'model.pt'
            print("SAVED MODEL", path)
            torch.save(model.state_dict(), path)
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(path.absolute())
            if G.model == 'autoencoder':
                model.save(G.logdir, batch[0])
            if G.eval_heavy:
                eval_heavy_time = time.time()
                eval_heavy(logger, model, test_ds, arbiter, G)
                logger['dt/eval_heavy'] = time.time() - eval_heavy_time
        logger = common.dump_logger(logger, writer, epoch, G)
        if epoch >= G.epochs:
            break


if __name__ == '__main__':
    model, train_ds, test_ds, arbiter, G = load_model_and_data()
    train(model, train_ds, test_ds, arbiter, G)
