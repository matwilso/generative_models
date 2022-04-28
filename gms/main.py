import time
import pathlib
import argparse
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from itertools import count
import torch as th
from gms import common
from gms import arbiters, autoregs, vaes, gans, diffusion
from gms.arbiters.autoencoder import Autoencoder
from gms.arbiters.classifier import Classifier
from gms.diffusion.diffusion import DiffusionModel
from gms.vaes.vae import VAE

# TRAINING SCRIPT

C = common.AttrDict()
C.model = 'vae'
C.bs = 64
C.hidden_size = 256
C.device = 'cuda'
C.num_epochs = 50
C.save_n = 100
C.logdir = pathlib.Path('./logs/')
C.arbiter_dir = pathlib.Path()
C.classifier_dir = pathlib.Path()
C.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
C.lr = 3e-4
C.class_cond = 0
C.binarize = 1
C.pad32 = 0

if __name__ == '__main__':
  # PARSE CMD LINE
  parser = argparse.ArgumentParser()
  for key, value in C.items():
    parser.add_argument(f'--{key}', type=common.args_type(value), default=value)
  tempC, _ = parser.parse_known_args()
  # SETUP
  Model = {
      'ae': Autoencoder,
      'classifier': Classifier,
      #'rnn': autoregs.rnn.RNN,
      #'made': autoregs.made.MADE,
      #'wavenet': autoregs.wavenet.Wavenet,
      #'pixelcnn': autoregs.pixelcnn.PixelCNN,
      #'gatedcnn': autoregs.gatedcnn.GatedPixelCNN,
      #'transformer': autoregs.transformer.TransformerCNN,
      'vae': VAE,
      #'vqvae': vaes.vqvae.VQVAE,
      #'gan': gans.gan.GAN,
      'diffusion': DiffusionModel,
  }[tempC.model]
  defaults = {'logdir': tempC.logdir / tempC.model}
  for key, value in Model.DC.items():
    defaults[key] = value
    if key not in tempC:
      parser.add_argument(f'--{key}', type=type(value), default=value)
  parser.set_defaults(**defaults)
  C = parser.parse_args()
  model = Model(C=C).to(C.device)
  writer = SummaryWriter(C.logdir)
  logger = common.dump_logger({}, writer, 0, C)
  train_ds, test_ds = common.load_mnist(C.bs, binarize=C.binarize, pad32=C.pad32)
  num_vars = common.count_vars(model)
  print('num_vars', num_vars)
  if str(C.arbiter_dir) == '.':
    arbiter = None
  else:
    arbiter = th.jit.load(C.arbiter_dir)
  if str(C.classifier_dir) == '.':
    classifier = None
  else:
    classifier = th.jit.load(C.classifier_dir)

  # TRAINING LOOP
  for epoch in count():
    # TRAIN
    train_time = time.time()
    for batch in train_ds:
      batch[0], batch[1] = batch[0].to(C.device), batch[1].to(C.device)
      # TODO: see if we can just use loss and write the gan such that it works.
      metrics = model.train_step(batch[0], batch[1])
      for key in metrics:
        logger[key] += [metrics[key].detach().cpu()]
    logger['dt/train'] = time.time() - train_time
    # TEST
    model.eval()
    with th.no_grad():
      # if we define an explicit loss function, use it to test how we do on the test set.
      if hasattr(model, 'loss'):
        for test_batch in test_ds:
          test_batch[0], test_batch[1] = test_batch[0].to(C.device), test_batch[1].to(C.device)
          test_loss, test_metrics = model.loss(test_batch[0], test_batch[1])
          for key in test_metrics:
            logger['test/' + key] += [test_metrics[key].detach().cpu()]
      else:
        test_batch = next(iter(test_ds))
        test_batch[0], test_batch[1] = test_batch[0].to(C.device), test_batch[1].to(C.device)
      # run the model specific evaluate function. usually draws samples and creates other relevant visualizations.
      eval_time = time.time()
      model.evaluate(writer, test_batch[0], test_batch[1], epoch, arbiter=arbiter, classifier=classifier)
      logger['dt/evaluate'] = time.time() - eval_time
    model.train()
    # LOGGING
    logger['num_vars'] = num_vars
    logger = common.dump_logger(logger, writer, epoch, C)
    if epoch % C.save_n == 0:
      path = C.logdir / 'model.pt'
      if C.model == 'ae' or C.model == 'classifier':
        jit_enc = th.jit.trace(model, test_batch[0])
        th.jit.save(jit_enc, str(path))
      else:
        th.save(model.state_dict(), path)
      print("SAVED MODEL", path)
    if epoch >= C.num_epochs:
      break
