# generative_models

Implementations of basic generative models for MNIST, along with descriptions using simple language.

Draws on lectures and demos from Berkeley [Deep Unsupervised Learning](https://sites.google.com/view/berkeley-cs294-158-sp20/) Course.
This repo is partially my notes from that class and partially meant to be a place to find simple implementations and clear explanations
of the fundamentals of deep generative modeling.

Many of these implementations will not scale far beyond MNIST, but they are just meant to represent the fundamental ideas
in a concise way. Or just because it's interesting to see what can be made to work well.

| Genereative models |  |  |  |
|-|-|-|-|
|  |  |  |  |
| Recurrent ||||
| RNN | this  | is | test |
| Masked |  |  |  |
| MADE | so | is | this |
| Wavenet | or | am | i |

<table style="width:100%">
  <tr>
    <th>Generative Models</th>
  </tr>
  <tr>
    <th>Autoregressive models</td>
  </tr>
  <tr>
    <td>RNN</td>
    <td>this</td>
    <td>are</td>
    <td>plots</td>
  </tr>
  <tr>
    <th>Masked models</td>
  </tr>
  <tr>
    <td>MADE</td>
    <td>this</td>
    <td>are</td>
    <td>plots</td>
  </tr>
</table>

## Autoregressive models

### Recurrent models
- [RNN/LSTM] trained to generate pixels one at a time

### Masked models
- [MADE] Masked Autoencoder for Distribution Estimation
- [Wavenet] Trained to generate pixels one at a time
- [PixelCNN (original)]
- [PixelCNN (gated/improved)]
- [TransformerCNN] Uses a transformer architecture where the individual pixels are considered tokens (28x28=784 of them for MNIST).

## Variational Autoencoders (VAES)

- [VAE (vanilla)]
- [VQ-VAE]

## GANs
- [GAN (vanilla)]. Somewhat based off DC-GAN, but scaled down to MNIST

## Future
- Flows
- Diffusion model
- EBM
// TODO: add class condition
// TODO: try interpolation
// TODO: visualizations.