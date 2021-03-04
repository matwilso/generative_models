# Generative Adversarial Networks (GANs)

GANs are pretty well covered elsewhere, due to their popularity, and the basic idea is pretty simple, so I just treat them briefly.

GANs were dominating in popularity for several years, because they are relatively inexpensive to train and sample from and they can be made to produce high quality visual samples. But recently likelihood based approaches have been giving them a run for their money, in recent work like GPT-3, VQ-VAE2, and DALL-E, for example.

The GAN loss tends to incentivize covering individual modes very well (but with mode collpase),
while the likelihood loss tends to incentivize covering all modes (but with blurriness)
Perhaps some mixture of the two losses is a good way forward.