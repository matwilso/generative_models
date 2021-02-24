# Variational AutoEncoder (VAE)

There are a few different ways to look at VAEs and understand them.

If we skip the derivation and just look at the final result of what they are.

There are several ways to look at VAEs. Skipping to the end and the connection with AEs makes
them easy to understand, but it makes the math and the reasoning and derivation harder to understand.
You can rationalize why certain components make sense, but you don't understand the probability and
approaches. Sometimes this can work, but it can also make further methods harder to understand.
If you don't understand the math.
The theory really isn't that bad. Once you understand it and get it into your brain, it is simple.
Getting it there is the hard part.


So we assume a graphical model. We think there is some latent variable z of the world.
Like maybe the scene we are looking at, the lighting, the camera angle, the bluriness.
And these z are going to generate some x, which is data like a specific image of a duck on a pond.

Now we want to evaluate the likelihood p(x) and be able to draw samples from it, like we did for the autoregressives.
For sampling, it's sample z then sample x given z. But how to sample z to be reasonable?

Likelihood = integral p(x|z) * p(z) dz.
Is intractable. You can't evaluate all values of z simulatneously.
If it's continuous, you can't evaluate this.

One thing you could is just take random samples over z. This is valid because that
integral is an expectation, and expectations can be replaced with sampling. However,
this produces noisy estimates.
Imageine we want to evaluate the likelihood of a duck photo,
how do we assure that we use the right latent that corresponds to ducks?
If we just sample, we might get lucky and be a duck. But the higher dimensionality you go, the harder this becomes.
So this ends up being a bad idea.

Using Bayes rule with our graphical model, there would be a way to analytically
compute the z that matches with a certain x. The posterior is: p(z|x) = p(x|z)/p(x) * p(z).
This tells you the exact z that matches the z.
But this requires evaluating the density of p(x), which we can't do.

There are a few possible paths to go in to fix this.

One is importance sampling. The trick of importance sampling is that you use some 
other distribution q(z) instead of p(z), which is hard to sample.
Going back to the prior sampling idea, And since we want it to depend on the x, so we can match up the ducks, we
we would want q(z) to account for that.
are going to have that be q(z|x). 
One thing I don't get about the notes is how you compute the KL of the actual posterior. I thought that was
intractible. And if we focus on a single xi. I guess we could just normalize. But still, how does that term expand?
I guess the intractible goes away if you do log prob? Maybe. Idk. skip probably.




You would expect that the z is unimodal if there is only one x that explains it.
But if you have multiple possible latents that describe the same x, for example
if you expect that there are multiple paths to reach the same x point, then you probably
want a multi-modal distribution.


What if we did REINFORCE on the VAE loss? Is this even possible?

approx_post(x).log_prob(z) * decoder(z).log_prob(x)
where z comes from samples.

The first term is the log_prob score. The second term is like the Advantage which we are using to modulate the gradients.
Make the z more likely in proportion to how likely it made the x.

Maybe it helps. It seems pretty weak. Is it any less weak than REINFORCE gradient is?
Likelihood ratio gradient.

Though you should probably use the output probability, not log prob. Then you subtract 0.5 to center it.
Bada boom bada bing.


The advantage of this would be that you don't need to be differentiable.
The score function or the sampling process.

Does VQ-VAE use this?










<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">


## Derivation

(The derivation and math are useful to understand if you want to extend them or
use them in other contexts. For example, the RSSM in Dreamer.)




## Latent variable models
The VAE is probably the most popular version of latent variable model.

A latent variable is something you never observe in the dataset, but which
you posit might exist as something that more compactly describes what we are seeing.
For example in MNIST, this might be the style of the digit being drawn, like whether
it is slanted, and how thick the lines are. These latent variables inform what you might
actually see.

Latent variable models have an advantage over autoregressive methods because
you can sample different parts independent of each, by conditioning on the latent variable.

You get a low-dimensional representation that may be useful for other tasks.
It is generally just easy to work with.

![](../../assets/vae_slide.png)


Advantages:
- sample different parts independently
- sample different parts independently


## Examples
- RIG
- Dreamer



# VQ-VAE