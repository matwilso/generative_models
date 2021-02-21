import argparse
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from itertools import count
import torch
import gms
from gms import utils

C = utils.AttrDict()
C.model = 'vae'
C.bs = 512
C.z_size = 128
C.hidden_size = 256
C.device = 'cuda'
C.log_n = 1000
C.done_n = 200
C.beta = 1.0
C.logdir = './logs/'
C.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
C.lr = 5e-4
C.class_cond = 0

if __name__ == '__main__':
  # PARSE CMD LINE
  parser = argparse.ArgumentParser()
  for key, value in C.items():
    parser.add_argument(f'--{key}', type=type(value), default=value)
  C = parser.parse_args()
  # SETUP
  model = {
    'vae': gms.VAE,
  }[C.model](C)
  model = model.to(C.device)
  writer = SummaryWriter(C.logdir)
  logger = utils.dump_logger({}, writer, 0, C)
  train_ds, test_ds = utils.load_mnist(C.bs)
  optimizer = Adam(model.parameters(), lr=C.lr)
  num_vars = utils.count_vars(model)

  # TRAINING LOOP 
  for epoch in count():
    # TRAIN
    for batch in train_ds:
      batch[0], batch[1] = batch[0].to(C.device), batch[1].to(C.device)
      optimizer.zero_grad()
      loss, metrics = model.loss(batch)
      loss.backward()
      optimizer.step()
      for key in metrics:
        logger[key] += [metrics[key].detach().cpu()]
    # TEST
    model.eval()
    with torch.no_grad():
      for test_batch in test_ds:
        test_batch[0], test_batch[1] = test_batch[0].to(C.device), test_batch[1].to(C.device)
        test_loss, test_metrics = model.loss(test_batch)
        for key in test_metrics:
          logger['test/' + key] += [test_metrics[key].detach().cpu()]
      model.evaluate(writer, test_batch, epoch)
    model.train()
    # LOGGING
    logger['num_vars'] = num_vars
    logger = utils.dump_logger(logger, writer, epoch, C)
    if epoch >= C.done_n:
      break