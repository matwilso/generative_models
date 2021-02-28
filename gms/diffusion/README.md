# Denoising Diffusion Probabilistic Models (aka Diffusion Models, aka DDPMs)

There are many related ideas, but the line can be traced to: https://arxiv.org/pdf/1503.03585.pdf.

I chose the Ho and followup Nichol+Dhariwal work because (1) produces great samples, and (2) produces great log-likelihoods.


This is kind of like flows, since your latents are in the same size as your data.
But you are using gaussian noise as the normalizing direction, and the reverse process is using learned function.
So this gives you more flexibility in networks. Also you can skip steps and your sampling can skip steps.
Also you can reuse that same network over again.

And because you are using gaussians, there is some math that says the reverse process can be described
in the same functional form (so theoretically, it should be possible to recover images by tracing back the guassian noise).


It's like this idea of chained probabilities.



We are going to describe by analogy with vaes.
Also in the graphical model, we are going to condition on the x0.
In the VAE case, the approx posterior was p(z|x) and we learned that.
In this case, it is a fixed process by which we just inject noise.

The latents are the x_T>1.



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

### What is "Langevin dynamics"?

It's basically using noisy SGD to create samples. you start with some initialization, then you run a few steps of gradient descent to get your sample. if you learn the gradient field instead of the original field, this would not require any backpropagation. which is the case for this paper.







###

This is a clever model. So you need a probabilstic model that is normalized.
You can either set it up like that, for example in autoregressive models or flows where this is 
assured. 

But the alternative approach is that you can define an energy model, where the distribution is unnormalized.
And then you call something the partition function Z for how to normalize it.

Understanding this actually requires a lot of knowledge of the statistical mechanics stuff.
That is where it comes from.

A diffusion process is: 


### What is diffusion?

Gases, people moving, 

https://en.wikipedia.org/wiki/Reverse_diffusion
reverse could be caused by osmosis.
oil and water separating.

flowing to areas of high concentration.

you can then think of the model as learning the reverse process.
these sometimes happen. learning the function that takes you from
to a level of reversion.

these type of reverse processes happen.
but you can imagine they take some energy or cost to make happen.
like it takes some effort.

in physics, you can think of it like phase separation.

and in information theory, you can think of it like going from higher noise
to low noise. and you can imagine this requires some work to be done.

What's even a more complicated but good example is osmosis vs. reverse osmosis.
Osmosis is basically diffusion. Then reverse osmosis, you pump pressure back in
to get rid of the stuff. Like purifying the drinking water, taking the contaminations
out of something. You can purify drinking water, but it takes a lot of cost
in pressure to push it back up the gradient.

We are learning a network to do that.

Yeah that is a good way to think about it. We are tainting the water,
and we are learning a network that can untaint it.





## Next section


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



























Cool how it is the same network applied over and over.
This makes me think it could be useful for thinking. You
just run this as many steps as you can, with an RNN to improve your thought.
You don't need to train it for many recurrent steps actually.
You can just run it at different steps, as is done in training.
Very cool. Actually extremely cool.
Wait holy shit.





I could take this idea and push on making something.
Like turning it into a paper idea.




You can define a very flexible probabilistic model by any function, you just have to have another function Z that can normalize it. Which can be quite hard to find. 
































The inspriation comes from something called non-equilirbium statistical physics. 
Statistical mechanics has many similarities with probabilsitic modeling, and in fact
things like Boltzman machines have been inspired by this stuff. It is a rich source of inspiration.
Interesting connection.


Non-equlibrium is to equlibrium what dynamics is to statics. It is more complicated,
because you now have moving parts. You take a statics course before you take a dynamics course.
You have to take care of their time courses.


Non-equillibrium thermo suggests brownian motion, which suggest langevin dynamics.



We assume there is going to be noise injected into our system, to unpurify it until
we eventually have no signal.



In brownian motion, there are noise and friction.
It may be useful to better understand this area.
Maybe at some point.

This actually seems like a pretty important area of math.
Statistical mechanics and probability. Nanotechnology and training
neural networks.

Sometimes I feel like you have to get really weird and go beyond the box
to come back and hit the right area. If your ideas are never crazy, you
are being too conservative.


Maximum entropy at thermodynamic equillibrium.
Learning how to reduce the entropy.




This refers to diffusion, like a gas diffusing from high concentration to low concentration.






First, they show an equivalence between denoising diffusion probabilistic models and denoising score matching.



It is nice that this process can go backwards and forwards very easily. Like you can inject noise a few steps and then go back out and do another sample and that is kind of interesting.



<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How to become expert at thing:<br>1 iteratively take on concrete projects and accomplish them depth wise, learning “on demand” (ie don’t learn bottom up breadth wise)<br>2 teach/summarize everything you learn in your own words<br>3 only compare yourself to younger you, never to others</p>&mdash; Andrej Karpathy (@karpathy) <a href="https://twitter.com/karpathy/status/1325154823856033793?ref_src=twsrc%5Etfw">November 7, 2020</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>


