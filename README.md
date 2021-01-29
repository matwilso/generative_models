# generative_models

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

// TODO: add class condition
// TODO: switch to MNIST
// TODO: work on autoreg stuff more
// TODO: try interpolation

Models:
- Autoregressive
  - MADE (RNN version + mask version)
  - Wavenet
  - PixelCNN/RNN, Wavenet, Transformer. 
- VAE
- GAN
- Flow
- Denoise
- EBM

Domains:
- Image
  - MNIST?
- Text
- Audio
- Motion
  - if you were to learn the model perfectly. such that you could optimize through it

  // TODO: refactor
  // TODO: switch everything over to mnist for now
  // TODO: add class conditioning