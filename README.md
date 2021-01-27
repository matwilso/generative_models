# generative_models

I wanted to study generative models, because I think they are important.
And then I stumbled across the USL course and so ended up roughly following htat.
So I learned a lot from those lectures and I borrow a lot.

What I do beyond that course is add extra notes in my own
words describing how the different things work. I also provide code in
a consistent format and easily modifiable and runnable, with nice plots
so you can study the behavior of these.

This is one repo that is easy to parse and provides a common way to look at many generative models.

Mainly this is for my own learning, but you may find it valuable.

My autoregressives is pretty thorough, heavily inspired by the great USL course.

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