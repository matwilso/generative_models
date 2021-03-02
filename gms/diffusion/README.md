# Diffusion Models

This implementation is based on [Denoising Diffusion Probabilistic Modeling (2020)](https://arxiv.org/pdf/2006.11239.pdf)
and [Improved Denoising Diffusion Probabilistic Modeling (2021)](https://arxiv.org/pdf/2102.09672.pdf), which build off
[Deep Unsupervised Learning using Nonequilibrium Thermodynamics (2015)](https://arxiv.org/pdf/1503.03585.pdf).

**Contents**
- [Analogies with other generative models](#analogies-with-other-generative-models)
  - [Flows](#flows)
  - [VAEs](#vaes)
  - [Autoregressive models](#autoregressive-models)
- [Terminology](#terminology)
  - [What is "diffusion"?](#what-is-diffusion)
  - [What is "score matching"?](#what-is-score-matching)
  - [What is "Langevin dynamics"?](#what-is-langevin-dynamics)
  - [Next section](#next-section)
- [Changes that Improved DDPMs introduces:](#changes-that-improved-ddpms-introduces)
- [Related papers](#related-papers)

## Analogies with other generative models
### Flows
[diagram of flows vs. diffusion models]

Diffusion models kind of look like flows.
For generating an image, you start with noise that is the same size of the image
and you pass it through several processing layers to gradually convert that noise into an image.

Other than that, they are pretty different. The forward "normalizing" process in diffusion models consists of gradually
adding small amounts of Gaussian noise until all signal is erased.
The reverse process uses a learned function to gradually remove Gaussian noise until a sample from the data distribution is produced. 

Diffusion models are not constrained to be bijective mappings; the forward process just adds
Gaussian noise and the reverse process is trained to undo that noise.
They also generally apply the same network over and over again for denoising, and they condition on a time index.
Also you can skip steps and your sampling can skip steps. 

### VAEs
[diagram of vaes vs. diffusion models, graphical model]
(indicate that arrows show the reverse diffusion direction.)

Diffusion models can be described by a directed graphical model and trained with a variational bound, like VAEs.
VAEs assume there are latent properties $z$ which underly the process that generates the data $x$.
The variational bound for VAEs looks like: 
$$L^{\text{VAE}}_{\text{VLB}} = \underbrace{-log p_\theta(x|z)}_\text{reconstruction} + \underbrace{KL(q_\phi(z|x) || p(z))}_{\sim \text{regularize encoder, make it possible to sample z}}$$

Diffusion models instead assume a Markov chain, whereby the latents
are the noisy intermediate versions of the data.
(You can imagine this corresponds to a physical diffusion process, where a gas
gradually diffuses in a room to reach a higher entropy state over time T.
If you could reverse that diffusion process, you could recover where
the gas particles started out (maybe they were in the shape of a smiley face).)
If we considered $x_{T}, x_{T-1}, ... x_{1}$ as the latents and used a multi-step graphical model,
the variational lower bound for this model looks somewhat similar to the VAE one:
$$L^{\text{Diffusion}}_{\text{VLB}} = \underbrace{-log p_\theta(x_0|x_1)}_{\text{reconstruction}} + \underbrace{\sum_{t=1}^{T} KL(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t))}_{\sim \text{inject noisy information from $x_0$ to train $p_\theta$}} + \underbrace{KL(q(x_T|x_0) || p(x_T))}_\text{negligible. $q(x_T|x_0)$ is pure noise}$$

In this case, we don't need to learn an approximate posterior $(q(x_{t-1}|x_t,x_0))$ because we can
compute it directly by a chain of Gaussian noise.

### Autoregressive models
[diagram of autoregressive property, or sampling process for autoregs and diffusion processes.]

Diffusion models have a notion of progressively producing a sample by running many forward
passes of the same network, and conditioning on prior generation.
However, it does not require you to define and respect an autoregressive ordering on the data.
The number of processing steps is not strictly dependent on the dimensionality of the data and can be chosen as a hyperparameter.
During sampling, you can reduce the number of processing steps. 

## Terminology

And because you are using gaussians, there is some math that says the reverse process can be described
in the same functional form (so theoretically, it should be possible to recover images by tracing back the guassian noise).
It's like this idea of chained probabilities.


### What is "diffusion"?

https://en.wikipedia.org/wiki/Diffusion

There are many analogies between statistical mechanics and the probability theory developed for that 
and machine learning systems (for example "temperatures" in sampling, and Boltzmann machines).
"Difussion" here refers to the physical process. You can imagine a concentrated
gas gradually diffusing in a room over time, until it reaches it's highest entropy state.
Over time, you gradually lose information about that gas.
This is the forward process of diffusion. It happens naturally. It is quite easy
to inject noise in a system. It is harder to reduce that noise. 

We can also imagine reverse diffusion processes, that can remove noise from a system, that can reduce it's entropy.
According to 2nd law of thermodynamics, it's impossible to reduce the entropy of a closed system.
So reverse diffusion requires spending external negative entropy (structure) to do this process.
Forward diffusion happens naturally. Reverse diffusion costs.
One example of a reverse diffusion process is in  [reverse osmosis water treatment plants](https://youtu.be/4RDA_B_dRQ0).
Polluted water is in a higher entropy state than clean water. To remove these pollutants requires lots of
energy to produce high-pressure gradients to pump water molecules through membranes that filter out pollutants.

In Denoising Diffusion Probabilistic Models, our forward process is Gaussian noise injection.
And we are going to spend optimization to learn a network that can reverse the noise process and give
us back clean samples.

Physically and information theory-wise, this seems interesting. You are learning a network
that can take a noise-shaped data and gradually create meaningful bits out of it.
Intuitively, this makes the learning problem easier because it's easy to
remove a bit of noise at a time, rather than all at once. Doing it this way, also let's us apply some
probabilistic math tricks that make the training more straightforward and let us specify log-likelihoods.

### What is "score matching"?

If we define our data distribution as p(x), then the score = grad_x log p(x). So in other words, the score is the slope
of the data distribution. 

You're familiar with training a model p_theta(x) to match an underlying p_data(x), using MLE.
This requires a normalized probability distribution. If your distribution is some unnormalized (like just
some output of an arbitrary neural network or energy function), you can write it like p_theta(x) = exp(-E(x)) / Z.
Where Z is the normalizing constant.

Z is hard to estimate and work with. But we can avoid having to learn it if instead of trying to learn p_data, we try to learn the score.
log(-E_theta(x)) - log(Z)
When we take the derivative of the distribution, then the normalizing constant goes away since it is not dependent on x.

This takes extra tricks and then sampling p(x) is a bit more complicated, but it can be done.
This leads to the next question:

### What is "Langevin dynamics"?

It's basically using noisy SGD to create samples. you start with some initialization, then you run a few steps of gradient descent to get your sample. if you learn the gradient field instead of the original field, this would not require any backpropagation. which is the case for this paper.

## Changes that Improved DDPMs introduces:
- learning the variance through that thing
- using hybrid = simple + 1e-4*vlb
- cosine schedule
- skipping during sampling.

## Related papers

- wavegrad
- song + ermon score matching, ddim.
