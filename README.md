# generative_models

**Implementations of fundamental deep generative models. (currently: Autoregressive models, VAEs, and GANs)**
>*"It is important to view knowledge as sort of a semantic tree -- make sure you understand the fundamental principles, ie the trunk and big branches, before you get into the leaves/details or there is nothing for them to hang on to"* - Elon

<!--, along with descriptions using simple language.-->

<!--probably ought to add some description of why i think generative models are important to understand-->

Partially these are my notes to understand the algorithms, made public to incentivize myself to be more thorough and make the code clean.
Beyond that, my goal is to provide a standardized repository for the important fundamental generative modeling algorithms,
with explanations and code that are as simple as possible to understand.

Parts of the code are taken from the Berkeley [Deep Unsupervised Learning](https://sites.google.com/view/berkeley-cs294-158-sp20/) Course.
Some of these implementations won't scale far beyond MNIST, because they are designed to represent the fundamental ideas very concisely.
(Most models are expressed within about 100 lines of code, including network architecture definitions.)
Some of them are super naive approaches, and it's just interesting to see how well they do on MNIST.

There is a central training script ([main.py](./gms/main.py)) that can load any of the models, train
them on MNIST, and log metrics to tensorboard. See usage below.
None of the performances here should be considered as hard evidence for or against an algorithm,
as they are not tuned.

**Contents**
- [Autoregressive models](#autoregressive-models)
- [Variational Autoencoders (VAEs)](#variational-autoencoders-vaes)
- [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
- [Future](#future)

**Install**
```
git clone https://github.com/matwilso/generative_models.git
cd generative_models/
pip install -e .
pip install -r requirements.txt
```

## Introduction

Generative models are a very important are of machine learning, that is going to be central
to the future of the field, fundamentally because they allow us to extract more useful bits from the environment.
And the cliched quote that "What I cannot create, I do not understand".

Over the years, we have developed several ways of using neural networks to generate data. 
You can break these into various classes, and each class faces various trade-offs and are useful in various settings.

It is unclear which is ultimately the most useful.
From 2015-2018, GANs were in the lead. But now I feel like
likelihood based approaches, including autoregressive models (mostly because of Transformers) and 
VAEs/VQVAEs have pulled ahead.
But who knows what might be useful from older approaches, Flows, and other things that
are just emerging or yet to be discovered.

We decouple the implementations from complex architectures, when possible.
The complex arches are important to understand. But also they add complexity to the core ideas.
And should be treated in some isolation.

## [Autoregressive models](gms/autoregs)

#### [RNN/LSTM](gms/autoregs/rnn.py)
Generate an MNIST image one pixel at a time with an LSTM
```
python main.py --model=rnn 
```
#### [MADE](gms/autoregs/made.py)
Run MADE on a flattened MNIST image
```
python main.py --model=made 
```
#### [Wavenet](gms/autoregs/wavenet.py)
Run a Wavenet on a flattened MNIST image
```
python main.py --model=wavenet 
```
#### [PixelCNN (original)](gms/autoregs/pixelcnn.py)
```
python main.py --model=pixelcnn 
```
#### [GatedPixelCNN (improved mask version)](gms/autoregs/gatedcnn.py)
```
python main.py --model=gatedcnn 
```
#### [TransformerCNN](gms/autoregs/transformer.py)
Kind of like a PixelCNN but uses a transformer architecture where the individual pixels are as considered tokens (28x28=784 of them for MNIST).
Kind of like ImageGPT.
```
python main.py --model=transformer 
```

## [Variational Autoencoders (VAEs)](gms/vaes/)

#### [VAE (vanilla)](gms/vaes/vae.py)
```
python main.py --model=vae 
```
#### [VQ-VAE](gms/vaes/vqvae.py)

The VQ-VAE is usually trained in a two-phase process. Phase 1 trains discrete encoder and decoder. Phase 2 trains
the prior that can produce the indexes of the latent codes, using a PixelCNN type approach.
Instead we train everything in a single Phase.
It's possible that this will lead to worse samples because the codes are constantly shifting so the PixelCNN has a harder time, but it simplifies
the code and lets you train it all in a single run.
And we also use our TransformerCNN, instead of our PixelCNN.

Also VQ-VAE usually produces codes in a 32x32 space, which is larger than an MNIST image lol.
We downsample to 7x7 codes, where K=64, so it is 64-way categorical. This still amounts 
to 64^49 possible values that the latent can take on. So you could say it's still pretty expressive.

```
python main.py --model=vqvae 
```
## [Generative Adversarial Networks (GANs)](gms/gans/)

GANs are a type of implicit model.

Implicit models work by basically training a discriminator of some sort.
For GAN, that is what it is. It could also be trained with contrastive learning.
Like in EBMs, where it happens that you use the same network for both.




#### [GAN (vanilla/scaled down DCGAN)](gms/gans/gan.py)
```
python main.py --model=gan 
```

## Future
- Diffusion model
- EBM
- Flows (not for awhile. i got sick of flows after messing with them a bit ago. i find they're messy and it's easy to get lost in the sauce)
- Non-generative self-supervised learning. contrastive.
- Score matching?

// TODO: more explanations of the algorithms <br>
// TODO: add class condition <br>
// TODO: try interpolation <br>
// TODO: visualizations of samples and training progression. <br>
// TODO: bits/dim for autoreg methods.  <br>
// TODO: FID or something for comparing sample qualities head to head. <br>
// TODO: head-to-head training times and such, both generally and specifically for autoregs, which have a very similar structure. <br>
// TODO: tests. <br>