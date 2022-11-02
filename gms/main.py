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

# cmd line args with their default values
CFG = common.AttrDict()
CFG.model = 'vae'
CFG.bs = 64
CFG.hidden_size = 256
CFG.device = 'cuda'
CFG.epochs = 50
CFG.save_n = 1
CFG.logdir = Path('./logs/')
CFG.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
CFG.lr = 3e-4
CFG.class_cond = 0
CFG.binarize = 1
CFG.pad32 = 0
CFG.mode = 'train'
CFG.weights_from = Path('.')
CFG.arbiter_dir = Path('.')
CFG.debug_eval = 0


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
            path = G.logdir / 'model.pt'
            print("SAVED MODEL", path)
            torch.save(model.state_dict(), path)
            if G.model == 'arbiter':
                model.save(G.logdir, batch[0])
        if epoch >= G.epochs:
            break


@torch.inference_mode()
def eval(model, train_ds, test_ds, G):
    assert G.arbiter_dir != Path('.'), "need to pass in arbiter dir"
    arbiter = torch.jit.load(G.arbiter_dir / 'arbiter.pt').to(G.device)
    if G.bs != 500:
        print("WARNING: YOU SHOULD USE A BS OF 500 FOR EVAL")
        # assert G.bs == 500, "do bs 500"
    fid_buffer = FID(num_features=64, feature_extractor=arbiter, device=G.device)

    # TEST
    model.eval()
    all_z_sample = []
    all_z_real = []

    for test_batch in test_ds:
        test_batch[0], test_batch[1] = test_batch[0].to(G.device), test_batch[1].to(
            G.device
        )
        samp = model.sample(test_batch[0].shape[0])
        fid_buffer.update((samp, test_batch[0]))
        z_samp = arbiter(samp)
        z_real = arbiter(test_batch[0])
        all_z_real.append(z_real)
        all_z_sample.append(z_samp)
        break
    # run the model specific evaluate function. usually draws samples and creates other relevant visualizations.
    fid_buff_out = fid_buffer.compute()
    print(f"{fid_buff_out = }")
    z_samp = z_samp
    z_real = z_real
    fid = common.compute_fid(z_samp.cpu().numpy(), z_real.cpu().numpy())
    precision, recall, f1 = map(
        lambda x: x.item(), common.precision_recall_f1(z_real, z_samp)
    )
    print(f"{fid = } {precision = } {recall = } {f1 = }")


def load_model_and_data():
    # PARSE CMD LINE
    parser = argparse.ArgumentParser()
    for key, value in CFG.items():
        parser.add_argument(f'--{key}', type=common.args_type(value), default=value)
    tempG, _ = parser.parse_known_args()
    # SETUP
    Model = common.discover_models()[tempG.model]
    defaults = {'logdir': tempG.logdir / tempG.model}
    for key, value in Model.DG.items():
        defaults[key] = value
        if key not in tempG:
            parser.add_argument(f'--{key}', type=type(value), default=value)

    if CFG.weights_from != Path('.'):
        loaded_hp_file = CFG.weights_from / 'hps.yaml'
        with open(loaded_hp_file) as f:
            loadedG = yaml.safe_load(f)
        for key, value in loadedG.items():
            defaults[key] = value

    parser.set_defaults(**defaults)
    G = parser.parse_args()
    model = Model(G=G).to(G.device)
    if G.weights_from != Path('.'):
        model.load_state_dict(
            torch.load(G.weights_from / 'model.pt', map_location=G.device)
        )
    train_ds, test_ds = common.load_mnist(G.bs, binarize=G.binarize, pad32=G.pad32)
    print('num_vars', common.count_vars(model))

    return model, train_ds, test_ds, G


if __name__ == '__main__':
    model, train_ds, test_ds, G = load_model_and_data()
    if G.mode == 'train':
        train(model, train_ds, test_ds, G)
    elif G.mode == 'eval':
        eval(model, train_ds, test_ds, G)
    else:
        raise Exception("unknown mode")
