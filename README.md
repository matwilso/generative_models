# generative_models

Implementations of basic generative models for MNIST, along with descriptions using simple language.

Draws on lectures and demos from Berkeley [Deep Unsupervised Learning](https://sites.google.com/view/berkeley-cs294-158-sp20/) Course.
This repo is partially my notes from that class and partially meant to be a place to find simple implementations and clear explanations
of the fundamentals of deep generative modeling.

Many of these implementations will not scale far beyond MNIST, but they are just meant to represent the fundamental ideas
in a concise way. Or just because it's interesting to see what can be made to work well.

## Autoregressive models

### Recurrent models
- [RNN/LSTM] trained to generate pixels one at a time

### Masked models
- [MADE] Masked Autoencoder for Distribution Estimation
- [Wavenet] Trained to generate pixels one at a time
- [PixelCNN (original)]
- [PixelCNN (gated/improved)]
- [TransformerCNN] Like a PixelCNN, but uses a transformer architecture where the individual pixels are as considered tokens (28x28=784 of them for MNIST).

## Variational Autoencoders ([VAEs](./vaes/))

### [VAE (vanilla)](./vaes/vae.py)
```
python main.py --model=vae --logdir=logs/vae/
```
### [VQ-VAE](./vaes/vqvae.py)

The VQ-VAE is usually trained in a two-phase process. Phase 1 trains discrete encoder and decoder. Phase 2 trains
the prior that can produce the indexes of the latent codes, using a PixelCNN type approach.
Instead we train everything in a single Phase.
I am pretty sure this leads to worse samples because the codes are constantly shifting and the PixelCNN is hard to learn, but it simplifies
the code and lets you train it in a single run.
And we also use our TransformerCNN, instead of our PixelCNN.

```
python main.py --model=vqvae --logdir=logs/vqvae/
```

## GANs
- [GAN (vanilla)]. Somewhat based off DC-GAN, but scaled down to MNIST

## Future
- Flows
- Diffusion model
- EBM
// TODO: add class condition
// TODO: try interpolation
// TODO: visualizations.