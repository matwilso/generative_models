from itertools import count
from pathlib import Path

import torch
import yaml
from ignite.metrics import FID
from torch.utils.tensorboard import SummaryWriter

from gms import common
from gms.train import load_model_and_data

# this script piggy backs off the train script for basic config, but is focused on more computationally intensive evaluation
# than you would want to include in the training loop, including FID score and PRC/F1


@torch.inference_mode()
def eval(model, test_ds, G):
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


if __name__ == '__main__':
    model, _, test_ds, G = load_model_and_data()
    eval(model, test_ds, G)
