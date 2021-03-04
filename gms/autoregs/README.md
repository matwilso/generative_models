# Autoregressive models

Autoregressive models require you to define and respect a temporal ordering on the data.
For example you may define an ordering for images that says top-left pixels come before bottom-right.
Then you must ensure that you never condition on any pixels that come *after* the current one---both
during training and sampling.

There are broadly two ways of achieving this: **RNNs** and **masking**.

With RNNs, you just feed the data points one-at-a-time and this ensures you never see the future. Pretty straighforward.

With masking, you don't rely on processing order; instead you use a feed-forward network
and mask our part of the architecture so that it can never see data from the future.

This folder contains several different approaches to masking. Each file handles the problem slightly differently.

As described in the Berkeley course lectures, generally you can categorize masks into two different types: A and B.
A-type mask means you set it up so you can only see your past (not yourself, so you can't cheat).
B-type mask means you set it up so you can only see yourself and your past.
As long as you mix this B-types with at least one A-type, autoregressive ordering will be respected.

<!--TODO: include my drawings and longer explanations -->

I would recommend stepping through the RNN approach, then Wavenet, then PixelCNN or TransformerCNN, to see how these models work and how masking is done.

<!--

## Explanations of specific approaches

### Wavenet

Wavenet is a pretty dumb way to deal with MNIST digits, but it is pretty easy and useful to understand.


### PixelCNN approaches


### Transformer

Flatten the image to 1D and then mask the Transformer such that it only sees the previous pixels.
(Only see pixels to the left and above).
-->