# generative_models

Implementations of basic generative models for MNIST, along with descriptions using simple language.

Draws on lectures and demos from Berkeley [Deep Unsupervised Learning](https://sites.google.com/view/berkeley-cs294-158-sp20/) Course.
This repo is partially my notes from that class and partially meant to be a place to find simple implementations and clear explanations
of the fundamentals of deep generative modeling.

Many of these implementations will not scale far beyond MNIST, but they are just meant to represent the fundamental ideas
in a concise way. Or just because it's interesting to see what can be made to work well.

- [Install](#install)
- [Autoregressive models](#autoregressive-models)
  - [RNN/LSTM](#rnnlstm)
  - [MADE](#made)
  - [Wavenet](#wavenet)
  - [PixelCNN (original)](#pixelcnn-original)
  - [GatedPixelCNN (improved mask version)](#gatedpixelcnn-improved-mask-version)
  - [TransformerCNN](#transformercnn)
- [Variational Autoencoders (VAEs)](#variational-autoencoders-vaes)
  - [VAE (vanilla)](#vae-vanilla)
  - [VQ-VAE](#vq-vae)
- [GANs](#gans)
  - [GAN (vanilla/scaled down DCGAN)](#gan-vanillascaled-down-dcgan)
- [Future](#future)

## Install
```
git clone https://github.com/matwilso/generative_models.git
cd generative_models/
pip install -e .
pip install -r requirements.txt
cd gms/
```

## [Autoregressive models](gms/autoregs)

### [RNN/LSTM](gms/autoregs/rnn.py)
```
python main.py --model=rnn 
```
### [MADE](gms/autoregs/made.py)
Masked Autoencoder for Distribution Estimation. I don't like MADE.
```
python main.py --model=made 
```
### [Wavenet](gms/autoregs/wavenet.py)
```
python main.py --model=wavenet 
```
### [PixelCNN (original)](gms/autoregs/pixelcnn.py)
```
python main.py --model=pixelcnn 
```
### [GatedPixelCNN (improved mask version)](gms/autoregs/gatedcnn.py)
```
python main.py --model=gatedcnn 
```
### [TransformerCNN](gms/autoregs/transformer.py)
Kind of like a PixelCNN but uses a transformer architecture where the individual pixels are as considered tokens (28x28=784 of them for MNIST).
Kind of like ImageGPT.
```
python main.py --model=transformer 
```

## [Variational Autoencoders (VAEs)](gms/vaes/)

### [VAE (vanilla)](gms/vaes/vae.py)
```
python main.py --model=vae 
```
### [VQ-VAE](gms/vaes/vqvae.py)

The VQ-VAE is usually trained in a two-phase process. Phase 1 trains discrete encoder and decoder. Phase 2 trains
the prior that can produce the indexes of the latent codes, using a PixelCNN type approach.
Instead we train everything in a single Phase.
This leads to worse samples because the codes are constantly shifting and the PixelCNN is hard to learn, but it simplifies
the code and lets you train it all in a single run.
And we also use our TransformerCNN, instead of our PixelCNN.

```
python main.py --model=vqvae 
```
## [GANs](gms/gans/)

### [GAN (vanilla/scaled down DCGAN)](gms/gans/gan.py)
```
python main.py --model=gan 
```

## Future
- Flows
- Diffusion model
- EBM
// TODO: more explanations of the algorithms
// TODO: add class condition
// TODO: try interpolation
// TODO: visualizations.