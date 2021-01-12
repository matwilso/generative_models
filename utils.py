import torch
import numpy as np

# TODO: make this a total data handler. load, device, everything

def export(img):
    img = (255 * (img.transpose(1,-1) + 1.0) / 2.0).detach().cpu().numpy().astype(np.uint8)
    img = img.reshape(-1, 32, 3)
    return img

def preproc(x):
    return (torch.as_tensor(x, dtype=torch.float32).transpose(1,-1) / 127.5) - 1.0

def sample_batch(train_data, bs):
    N = train_data.data.shape[0]
    idxs = np.random.randint(0, N, bs)
    x = train_data.data[idxs]
    return preproc(x)

