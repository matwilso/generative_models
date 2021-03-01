# Denoising Diffusion Probabilistic Models (aka Diffusion Models, aka DDPMs)

There are many related ideas, but the line can be traced to: https://arxiv.org/pdf/1503.03585.pdf.

I chose the Ho and followup Nichol+Dhariwal work because (1) produces great samples, and (2) produces great log-likelihoods.


## Analogies with Flows and VAEs

**Flows.**
This is kind of like flows, since your latents are in the same size as your data.
But you are using gaussian noise as the normalizing direction, and the reverse process is using learned function.
So this gives you more flexibility in networks. Also you can skip steps and your sampling can skip steps.
Also you can reuse that same network over again.

And because you are using gaussians, there is some math that says the reverse process can be described
in the same functional form (so theoretically, it should be possible to recover images by tracing back the guassian noise).

It's like this idea of chained probabilities.

**VAEs.**
It's also kind of like a VAE.
We are going to describe by analogy with vaes.
Also in the graphical model, we are going to condition on the x0.
In the VAE case, the approx posterior was p(z|x) and we learned that.
In this case, it is a fixed process by which we just inject noise.
Since it is same sized as the data.

So like a way, but there are many intermediary steps and the latent is same sized.


The latents are the x_T>1.


**Autoregs.**

There is a notion of progressively sampling. But the number of steps
is not dependent on the dimensionality of the data. So you can set a number
and in practice, make it not very many steps.
This iterative process seems pretty powerful.




Cool how it is the same network applied over and over.
This makes me think it could be useful for thinking. You
just run this as many steps as you can, with an RNN to improve your thought.
You don't need to train it for many recurrent steps actually.
You can just run it at different steps, as is done in training.
Very cool. Actually extremely cool.
Wait holy shit.


## Changes that Improved DDPMs introduces:
- learning the variance through that thing
- using hybrid = simple + 1e-4*vlb
- cosine schedule
- skipping during sampling.



## Terminology

### What is "diffusion"?

https://en.wikipedia.org/wiki/Diffusion

This work takes inspiration from physics. You can imagine pouring a bunch of different chemicals
in a vat of water and they will all mix together and become a mixture with highest entropy.
For information, you can imagine transmitting data along a noisy analog channel and it
will get increasingly noisy the farther it travels.
This is a natural process. Things mix and get noisier. It's easy to add noise, harder to take it away.
This is the forward diffusion process.

We can also imagine a reverse diffusion process, something that removes noise, that separates out
clean water from all of the chemicals that we added. But we know that we are going to have to put in work 
to battle the entropy and clean up the signal. In reverse osmosis water treatment plants, we have to use
high-pressure systems to pump water molecules through a semi-permeable membrane. We can climb up the entropy
gradient, by using external effort. It costs energy to do.
In this work, we are going to learn a network that does this work for us, and we are going to
spend optimization to get something that can take noise/polutions to the data and undo them to clean
it up.

https://youtu.be/4RDA_B_dRQ0

And roughly the way we are going to do this is by taking a data sample, adding noise to it, and
training our network to undo the noise. 

And specifically, we are going to use a markov chain graphical model, where small amounts of noise
are added over many steps. Intuitively, this makes the learning problem easier because it's easy to
remove a bit of noise at a time, rather than all at once. Doing it this way, also let's us apply some
probabilistic math tricks that make the training more straightforward and let us specify log-likelihoods.

Physically and information theory-wise, this seems interesting. You are learning a network
that can take a noise-shaped data and gradually create meaningful bits out of it.

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

### Next section

Now imagine you are trying to generate a sample p(x0).
How do you do that?


Ok they use importance sampling, as does the abbeel vae explaantion.

We can't use the p to calculate this during training, since it won't correspond to
good reverse diffusion. So we are going to use a MC version by using the .

We can't sample from it because it requires knowing the reverse diffusion.
But we have the forward process we can use.


This is similar to the issue we have in VAEs, where if 



So you've got forward diffusion, you're trying to learn the reverse diffusion.
How you can do that is you taint the data and then try to untaint it.






## Related papers

- 15
- ho 
- unixpickle
- wavegrad
- song + ermon score matching, ddim.
