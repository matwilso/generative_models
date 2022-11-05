import torch.nn.functional as F
from einops import rearrange, repeat
from torch.optim import Adam

from gms import common
from gms.arbiters.autoencoder import Encoder


class Classifier(common.GM):
    DG = common.AttrDict()  # default G
    DG.eval_heavy = False

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
        preds = self.net(x[:8]).argmax(1)
        correct_mask = preds == y[:8]
        # we have to clone because einops does this bullshit where it creates a view instead of actually duplicating the rows. goddamn.
        # TODO: file a bug report
        imgs = repeat(x[:8].cpu(), 'b c h w -> b (repeat c) h w', repeat=3).clone()
        imgs[correct_mask, 0] = 0
        imgs[correct_mask, 2] = 0
        imgs[~correct_mask, 1] = 0
        imgs[~correct_mask, 2] = 0
        writer.add_image(
            'reconstruction', rearrange(imgs, 'n c h w -> c h (n w)'), epoch
        )
