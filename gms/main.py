import pathlib
import argparse
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from itertools import count
import torch
import gms
from gms import utils

# TRAINING SCRIPT

C = utils.AttrDict()
C.model = 'vae'
C.bs = 64
C.hidden_size = 256
C.device = 'cuda'
C.done_n = 200
C.save_n = 5
C.beta = 1.0
C.logdir = pathlib.Path('./logs/')
C.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
C.lr = 3e-4
C.class_cond = 0
C.binarize = 1

if __name__ == '__main__':
  # PARSE CMD LINE
  parser = argparse.ArgumentParser()
  for key, value in C.items():
    parser.add_argument(f'--{key}', type=utils.args_type(value), default=value)
  tempC, _ = parser.parse_known_args()
  # SETUP
  Model = {
    'vae': gms.VAE,
    'vqvae': gms.VQVAE,
    'gan': gms.GAN,
    'transformer': gms.TransformerCNN,
  }[tempC.model]
  defaults = {}
  for key, value in Model.DC.items():
    defaults[key] = value
    if key not in tempC:
      parser.add_argument(f'--{key}', type=type(value), default=value)
  parser.set_defaults(**defaults)
  C = parser.parse_args()
  model = Model(C=C).to(C.device)
  writer = SummaryWriter(C.logdir)
  logger = utils.dump_logger({}, writer, 0, C)
  train_ds, test_ds = utils.load_mnist(C.bs, binarize=C.binarize)
  num_vars = utils.count_vars(model)

  # TRAINING LOOP 
  for epoch in count():
    # TRAIN
    for batch in train_ds:
      batch[0], batch[1] = batch[0].to(C.device), batch[1].to(C.device)
      # TODO: see if we can just use loss and write the gan such that it works.
      metrics = model.train_step(batch)
      for key in metrics:
        logger[C.model+'/'+key] += [metrics[key].detach().cpu()]
    # TEST
    model.eval()
    with torch.no_grad():
      for test_batch in test_ds:
        test_batch[0], test_batch[1] = test_batch[0].to(C.device), test_batch[1].to(C.device)
        test_loss, test_metrics = model.loss(test_batch)
        for key in test_metrics:
          logger[C.model+'/test/' + key] += [test_metrics[key].detach().cpu()]
      model.evaluate(writer, test_batch, epoch)
    model.train()
    # LOGGING
    logger['num_vars'] = num_vars
    logger = utils.dump_logger(logger, writer, epoch, C)
    if epoch % C.save_n == 0:
      path = C.logdir / 'model.pt'
      print("SAVED MODEL", path)
      torch.save(model.state_dict(), path)
    if epoch >= C.done_n:
      break