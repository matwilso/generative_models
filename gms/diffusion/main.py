from gms.diffusion.diffuser import GaussianDiffuser
import sys
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from basicnet import BasicNet
from gms import utils

C = utils.AttrDict()
C.bs = 32
C.z_size = 128
C.bn = 0
C.device = 'cuda'
C.log_n = 1000
C.done_n = 20
C.b = 0.1
C.logdir = './logs/'
C.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
C.lr = 1e-3
C.class_cond = 0
C.hidden_size = 512
C.append_loc = 1
C.overfit_batch = 0
C.n_layer = 2
C.n_head = 4
C.n_embed = 128
C.block_size = 28*28


if __name__ == '__main__':
  # TODO: -1,1 center the data
  C = utils.parseC(C)
  writer = SummaryWriter(C.logdir)
  logger = utils.dump_logger({}, writer, 0, C)
  train_ds, test_ds = utils.load_mnist(C.bs)
  _batch = next(iter(train_ds))
  _batch[0] = _batch[0].to(C.device)
  model = BasicNet(C).to(C.device)
  optimizer = Adam(model.parameters(), lr=C.lr)
  out = model.forward(_batch[0])

  betas = np.linspace(1e-4, 0.01, 500)
  diffuser = GaussianDiffuser(betas, model_mean_type='eps', model_var_type='fixedlarge', loss_type='mse')
  import ipdb; ipdb.set_trace()