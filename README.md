# generative_models

Implementations of fundamental deep generative models.
<!--, along with descriptions using simple language.-->

Partially these are my notes to understand the algorithms and I make them public to incentivize myself to understand them clearly and make the code clean.
Beyond that, my goal is to provide a central source of many of the important fundamental generative models, with code that is as simple as possible to understand.

Parts of the code are taken from the Berkeley [Deep Unsupervised Learning](https://sites.google.com/view/berkeley-cs294-158-sp20/) Course,
which I have been following. Many of these implementations will not scale far beyond MNIST, but they are just meant to represent the fundamental ideas
in a concise way. And sometimes it's just interesting to see what can be made to work well.

I have a central training script ([main.py](./gms/main.py)) that can load any of the models, train
them on MNIST, and log metrics to tensorboard.


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

## [Autoregressive models](gms/autoregs)

#### [RNN/LSTM](gms/autoregs/rnn.py)
```
python main.py --model=rnn 
```
#### [MADE](gms/autoregs/made.py)
```
python main.py --model=made 
```
#### [Wavenet](gms/autoregs/wavenet.py)
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
This leads to worse samples because the codes are constantly shifting and the PixelCNN is hard to learn, but it simplifies
the code and lets you train it all in a single run.
And we also use our TransformerCNN, instead of our PixelCNN.

Also VQ-VAE usually produces codes in a 32x32 space, which is larger than an MNIST image lol.
We downsample to 7x7 codes, where K=64, so it is 64-way categorical. This still amounts 
to 64^49 possible values that the codes can take on. So you could say it's still pretty expressive.

```
python main.py --model=vqvae 
```
## [Generative Adversarial Networks (GANs)](gms/gans/)

#### [GAN (vanilla/scaled down DCGAN)](gms/gans/gan.py)
```
python main.py --model=gan 
```

## Future
- Flows
- Diffusion model
- EBM

// TODO: more explanations of the algorithms <br>
// TODO: add class condition <br>
// TODO: try interpolation <br>
// TODO: visualizations. <br>
// TODO: bits/dim for autoreg methods.  <br>
// TODO: FID or something for comparing sample qualities head to head. <br>
// TODO: head-to-head training times and such, both generally and specifically for autoregs, which have a very similar structure. <br>
// TODO: tests. <br>