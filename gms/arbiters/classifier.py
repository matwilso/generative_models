import torch.nn.functional as F
from einops import rearrange, repeat
from torch.optim import Adam

from gms import common
from gms.arbiters.autoencoder import Encoder


class Classifier(common.GM):
    DG = common.AttrDict()  # default G
    DG.eval_heavy = False
    DG.epochs = 6  # we start to overfit after about this many
    DG.binarize = 0
    DG.save_n = 1

    def __init__(self, G):
        super().__init__(G)
        self.net = Encoder(10, G)
        self.optimizer = Adam(self.parameters(), lr=G.lr)

    def forward(self, x):
        return self.net(x)

    def loss(self, x, y):
        z = self.net(x)
        loss = F.cross_entropy(z, y)
        metrics = {'cross_entropy_loss': loss}
        return loss, metrics

    def evaluate(self, writer, x, y, epoch, arbiter=None, classifier=None):
        """run samples and other evaluations"""
        N = 10
        preds = self.net(x[:N]).argmax(1)
        correct_mask = preds == y[:N]
        # need to clone because einops repeat does a view instead of a copy
        imgs = repeat(x[:N].cpu(), 'b c h w -> b (repeat c) h w', repeat=3).clone()
        # green for correct, red for incorrect
        imgs[correct_mask, 0] = 0
        imgs[correct_mask, 2] = 0
        imgs[~correct_mask, 1] = 0
        imgs[~correct_mask, 2] = 0
        writer.add_image(
            'classifier/pred', rearrange(imgs, 'n c h w -> c h (n w)'), epoch
        )
