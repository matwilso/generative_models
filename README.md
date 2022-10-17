# generative_models

**[Motivation](#motivation) | [Autoregressive models 📃](#autoregressive-models) | [VAEs 🪞](#variational-autoencoders-vaes) | [GANs 🧑‍🎨/🕵](#generative-adversarial-networks-gans) | [Diffusion models 🧪](#diffusion-models)**

The goal of this repo is to provide implementations of the important fundamental approaches to generative modeling, with explanations and code that are as simple as possible to understand (in pytorch).
Most models are expressed within about 100 lines of code, including network architecture definitions---except for diffusion models, which get a bit messy 😅.
As such, they do not contain all SOTA architectures or other tricks.

There is a central training script ([main.py](./gms/main.py)) that can load any of the models, train
them on MNIST, and log metrics to tensorboard. See usage below.

**Install**
```
git clone https://github.com/matwilso/generative_models.git
cd generative_models/
pip install -e .
pip install -r requirements.txt
```

## Motivation

>*"What I cannot create I do not understand"* - Feynman

>*"It is important to view knowledge as sort of a semantic tree -- make sure you understand the fundamental principles, ie the trunk and big branches, before you get into the leaves/details or there is nothing for them to hang on to"* - Elon

Unsupervised learning algorithms can extract many more bits from the environment than supervised learning and reinforcement learning can,
which makes them much better suited for training larger and more powerful neural networks (a la the LeCake argument).
Unsupervised learning is thus going to be a major driving force behind progress in all AI fields ([robot learning](https://matwilso.github.io/robot-future/), for example), and is worth studying and becoming an expert on.

There are many specific approaches to unsupervised learning and generative modeling specifically, and each face different trade-offs.
This repo offers some intuitive explanations, and simple as possible code for demonstrating and experimenting with generative models specifically on MNIST digits.  For further resources, I would suggest the [Deep Unsupervised Learning Berkeley Course](https://sites.google.com/view/berkeley-cs294-158-sp20/) and the [Deep Learning Textbook (chapters 15-20)](https://www.deeplearningbook.org/).
(Parts of the autoregressive code are based on demos from the Berkeley course. Other parts of the code
are based on various repos on the internet, which in turn are based on further upstream sources, and I provide links where relevant.
The main thing I aim for is concise and easy to understand code, and for standardization across algorithms to the extent possible.
My [VQ-VAE](./gms/vaes/vqvae.py) and [Diffusion models](./gms/diffusion/diffusion.py)
are especially simple compared to implementations I have seen of them online.)

<!--
, so it is important
to understand the fundamental approaches.

, that is going to be central
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
-->

## [Autoregressive models](gms/autoregs)

#### [RNN/LSTM](gms/autoregs/rnn.py)
Generate an MNIST image one pixel at a time with an LSTM
```
python -m gms.main --model=rnn
```
#### [MADE](gms/autoregs/made.py)
Run MADE on a flattened MNIST image
```
python -m gms.main --model=made
```
#### [Wavenet](gms/autoregs/wavenet.py)
Run a Wavenet on a flattened MNIST image
```
python -m gms.main --model=wavenet
```
#### [PixelCNN (original)](gms/autoregs/pixelcnn.py)
```
python -m gms.main --model=pixelcnn
```
#### [GatedPixelCNN (improved mask version)](gms/autoregs/gatedcnn.py)
```
python -m gms.main --model=gatedcnn
```
#### [TransformerCNN](gms/autoregs/transformer.py)
Kind of like a PixelCNN but uses a transformer architecture where the individual pixels are as considered tokens (28x28=784 of them for MNIST).
Kind of like ImageGPT.
```
python -m gms.main --model=transformer
```

## [Variational Autoencoders (VAEs)](gms/vaes/)

#### [VAE (vanilla)](gms/vaes/vae.py)
```
python -m gms.main --model=vae
```
#### [VQ-VAE](gms/vaes/vqvae.py)

The VQ-VAE is usually trained in a two-phase process. Phase 1 trains discrete encoder and decoder. Phase 2 trains
the prior that can produce the indexes of the latent codes, using a PixelCNN type approach.
Instead we train everything in a single Phase.
It's possible that this will lead to worse samples because the codes are constantly shifting so the PixelCNN has a harder time, but it simplifies
the code and lets you train it all in a single run.
And we also use our TransformerCNN, instead of our PixelCNN.

Also VQ-VAE usually produces codes in a 32x32 space, which is larger than an MNIST image lol.
We instead downsample to 7x7 codes, which are 64-way categorical (K=64). This space still amounts 
to 64^49 possible values that the latent can take on. So still pretty expressive.

```
python -m gms.main --model=vqvae
```
## [Generative Adversarial Networks (GANs)](gms/gans/)

#### [GAN (vanilla/scaled down DCGAN)](gms/gans/gan.py)
```
python -m gms.main --model=gan
```

## [Diffusion Models](gms/diffusion/)

```
python -m gms.main --model=diffusion
```

(after 10 epochs of training. left: sampling process (x_500, x_499, x_498, ..., x_0), right: predictions of x_0, given current x_t.)

![](assets/diffusion_sample_10.gif)
![](assets/diffusion_10.gif)


## Future
- EBMs. Not used in any recent SOTA results afaik, but knowing about EBMs seems to unite a few other approaches like GANs and Diffusion models in my mind. 
- Flows (eh might just skip them. i got sick of flows after messing with them a bit ago. i find them quite messy in practice. plus not much seems to have happened with them. they're too limiting in constraints they place on architecture it seems.)
- Non-generative self-supervised learning. contrastive.
- Score matching? seems quite related to diffusion models

// TODO: more explanations of the algorithms <br>
// TODO: add class condition <br>
// TODO: try interpolation <br>
// TODO: visualizations of samples and training progression. <br>
// TODO: bits/dim for autoreg methods.  <br>
// TODO: FID or something for comparing sample qualities head to head. <br>
// TODO: head-to-head training times and such, both generally and specifically for autoregs, which have a very similar structure. <br>
// TODO: tests. <br>
