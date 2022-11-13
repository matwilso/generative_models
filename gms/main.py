import argparse
import time
from itertools import count
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from ignite.metrics import FID
from torch.utils.tensorboard import SummaryWriter

from gms import common

# TRAINING SCRIPT

# cmd line args with their default values. models can add additional args of their own
# DG = (D)efault G, where G is a config object. Single letters C, H, F were already taken, so G. lol
# TODO: make this support default None values, would require rework. maybe like dataclass
DG = common.AttrDict()
DG.model = 'vae'
DG.bs = 64
DG.hidden_size = 256
DG.device = 'cuda'
DG.epochs = 50
DG.save_n = 5
DG.logdir = Path('./logs/')
DG.lr = 3e-4
DG.class_cond = 0
DG.binarize = 1
DG.pad32 = 0
DG.mode = 'train'
# path to a model that you want to load and keep training
DG.weights_from = Path('.')
# for computing a latent space for eval metrics
DG.autoencoder = Path('./weights/autoencoder.pt')
# for conditional generation eval metrics
DG.classifier = Path('./weights/classifier.pt')
DG.eval_heavy = 0
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
            loadedG = common.AttrDict(yaml.load(f, Loader=yaml.Loader))
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

    if 'full_cmd' in defaults:
        defaults.pop('full_cmd')
    # do the final parse of cmd line args for what user passes in and instantiate everything
    parser.set_defaults(**defaults)
    G = common.AttrDict(parser.parse_args().__dict__)
    model = Model(G=G).to(G.device)
    if G.weights_from != Path('.'):
        model.load_state_dict(
            torch.load(G.weights_from, map_location=G.device), strict=False
        )
    train_ds, test_ds = common.load_mnist(G.bs, binarize=G.binarize, pad32=G.pad32)
    print('num_vars', common.count_vars(model))
    autoencoder = torch.jit.load(G.autoencoder).to(G.device) if G.eval_heavy else None
    classifier = (
        torch.jit.load(G.classifier).to(G.device)
        if G.eval_heavy and G.class_cond
        else None
    )

    return model, train_ds, test_ds, autoencoder, classifier, G


@torch.inference_mode()
def eval_heavy(logger, model, test_ds, autoencoder, classifier, G):
    """
    More computationally intensive eval that draws many samples from the generative model and computes
    FID and precision/recall metrics.
    """
    TOTAL_SAMPLES = 500  # beyond this, we started getting diminishing returns

    # sample the model and get latent vectors for samples as well as test set examples
    # so that we can then compare the two distributions (sampled vs. test).
    sample_ct = 0
    all_z_sample = []
    all_z_real = []
    fid_buffer = FID(num_features=64, feature_extractor=autoencoder, device=G.device)
    metrics = {}

    # aggregate latent vectors
    if G.class_cond:
        metrics['classifier_loss'] = []
        all_z_cond_sample = []
    for test_batch in test_ds:
        test_x, test_y = test_batch[0].to(G.device), test_batch[1].to(G.device)
        bs = test_x.shape[0]
        if G.class_cond:
            # generate class conditioned sample and evaluate it with the classifier
            cond_samp = model.sample(bs, y=test_y)
            preds = classifier(cond_samp)
            metrics['classifier_loss'].append(F.cross_entropy(preds, test_y).item())
            all_z_cond_sample.append(autoencoder(cond_samp))

        samp = model.sample(bs, y=-torch.ones_like(test_y))
        fid_buffer.update((samp, test_x))
        all_z_real.append(autoencoder(test_x))
        all_z_sample.append(autoencoder(samp))
        sample_ct += bs
        if sample_ct >= TOTAL_SAMPLES:
            break

    # evaluate metrics
    metrics['ignite_fid'] = fid_buffer.compute()
    z_samp = torch.cat(all_z_sample)
    z_real = torch.cat(all_z_real)
    metrics['fid'] = common.compute_fid(z_samp.cpu().numpy(), z_real.cpu().numpy())
    metrics.update(common.precision_recall_f1(real=z_real, gen=z_samp))
    if G.class_cond:
        z_cond_samp = torch.cat(all_z_cond_sample)
        cond_metrics = common.precision_recall_f1(real=z_real, gen=z_cond_samp)
        cond_metrics['fid'] = common.compute_fid(
            z_cond_samp.cpu().numpy(), z_real.cpu().numpy()
        )
        metrics.update(common.prefix_dict('cond_', cond_metrics))
        # metrics.update(common.prefix_dict('class_conditioned.', cond_metrics))

    for key, val in metrics.items():
        logger[f'eval/{key}'] += [np.mean(common.to_numpy(val))]


def train(model, train_ds, test_ds, autoencoder, classifier, G):
    writer = SummaryWriter(G.logdir)
    logger = common.dump_logger({}, writer, 0, G)

    # TRAINING LOOP
    for epoch in count(1):
        # TRAIN
        train_time = time.time()
        for batch in train_ds:
            if G.skip_training:
                break
            train_x, train_y = batch[0].to(G.device), batch[1].to(G.device)
            # TODO: see if we can just use loss and write the gan such that it works.
            metrics = model.train_step(train_x, train_y)
            for key in metrics:
                logger[f'{G.model}/train/{key}'] += [metrics[key].detach().cpu()]

        logger['dt/train'] = time.time() - train_time
        # TEST
        model.eval()
        with torch.no_grad():
            # if we define an explicit loss function, use it to test how we do on the test set.
            if hasattr(model, 'loss'):
                for test_batch in test_ds:
                    test_x, test_y = test_batch[0].to(G.device), test_batch[1].to(
                        G.device
                    )
                    _, test_metrics = model.loss(test_x, test_y)
                    for key in test_metrics:
                        logger[f'{G.model}/test/{key}'] += [
                            test_metrics[key].detach().cpu()
                        ]
            else:
                test_batch = next(iter(test_ds))
                test_x, test_y = test_batch[0].to(G.device), test_batch[1].to(G.device)
            # run the model specific evaluate function. usually draws samples and creates other relevant visualizations.
            eval_time = time.time()
            model.evaluate(writer, test_x, test_y, epoch)
            logger['dt/eval'] = time.time() - eval_time
        # LOGGING
        logger['num_vars'] = common.count_vars(model)
        if epoch % G.save_n == 0:
            model.save(G.logdir, test_x, test_y)
            print("SAVED MODEL", G.logdir)

            if G.eval_heavy:
                print("RUNNING HEAVY EVAL...")
                eval_heavy_time = time.time()
                eval_heavy(logger, model, test_ds, autoencoder, classifier, G)
                logger['dt/eval_heavy'] = time.time() - eval_heavy_time
                print("DONE HEAVY EVAL")
        logger = common.dump_logger(logger, writer, epoch, G)
        if epoch >= G.epochs:
            break
        model.train()


if __name__ == '__main__':
    model, train_ds, test_ds, autoencoder, classifier, G = load_model_and_data()
    train(model, train_ds, test_ds, autoencoder, classifier, G)
