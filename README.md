# generative_models

The goal of this is to (1) learn about generative models and keep a public version for motivation
for accuracy and up-to-dateness (also notes in my own words for studying USL course), (2) high-quality explanations of the fundamentals along with basic
implementations that generate good MNIST samples. The aim is for the simplest possible version
that solves MNIST and is fairly robust to hyperparameters.

Explaining in deep detail the ideas behind each algorithm, and explanations of all
the details. Understanding why every single line of code is there. The theory and everything.

My general resource for this was the USL course.
Where it makes sense, I borrow from some of the code provided from the course demos.



Implementing and digging into code is very expensive.
But it's really important to know what exists and to be familiar
with the details so that you have a complete enough toolbox to solve
your problems. I believe generative modeling is an important set of tools
for many future aspects of deep learning. Because they are what lets us do unsupervised
learning and understand the world. The old quote that is super cliche and over-used
at this point: "What I cannot create, I do not understand."

"Know how to solve every problem that has been solved."




## Models

Models:
- Autoregressive
  - MADE (RNN version + mask version)
  - Wavenet
  - PixelCNN/RNN, Wavenet, Transformer. 
- VAEs
- GAN

Autoregressive the math and intuition is pretty easy and the architectures and the details are a bit harder to understand.
VAEs the math is harder to understand. Some of the intuition is easy, if you just see it as extension of an
autoencoder. But the theory behind it requires more knowledge of probability. In the end though, the details
are pretty simple. The base architecture and training process is extremely straightforward.
For GANs, the basic intuition and math are pretty easy, and so is the architecture. It's just pretty
hard to get the damn thing working. They are fickle and require special details to work well.

## Future
- Flows
- Diffusion 
- EBM
// TODO: add class condition
// TODO: try interpolation





I wanted to study generative models, because I think they are important.
And then I stumbled across the USL course and so ended up roughly following htat.
So I learned a lot from those lectures and I borrow a lot.

What I do beyond that course is add extra notes in my own
words describing how the different things work. I also provide code in
a consistent format and easily modifiable and runnable, with nice plots
so you can study the behavior of these.
I thought I have pretty good understanding and notes of these and their specifics.

This is one repo that is easy to parse and provides a common way to look at many generative models.

Mainly this is for my own learning, but you may find it valuable.

One way you may is by going through and dropping in on the code and seeing what it does.

This is basically trying to understand all generative models.
And playing with the code and going through it all seems the best way to really understand it.
If I didn't write the code, I need to understand the decision behind every single line.
And I have commented anything that seemed non-obvious to me.

My autoregressives is pretty thorough, heavily inspired by the great USL course.

You should be able to clone this and run it with minimal effort.
I've listed all dependencies and how to install then.
If you have trouble, do let me know.

This ids 

It's nice to see all these separate methods in a single good framework. 

[individual lectures]
[lecture that compares trade-offs between everything. and i might summarize it here]

The thing about this is that scaling up is a lot of the effort.
Like you can make things work on MNIST fairly easily. But generally
those methods fail when you scale up to a harder problem. 
This is about fundamentals.