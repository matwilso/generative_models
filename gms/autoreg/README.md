# Autoregressive Models

// TODO: make these all consistent in here and get rid of duplicate code. just because it's a pain to deal with.
// they are all way simple, so it should be fine.

# RNN version

// todo: make pixelrnn if it's not too bad

# Masking

The central idea of "masking" is that you are going to design your structure such
you can only look at the past and never yourself or the future.

This makes sense if you think about the sampling process.
You are going to start with a zero-image.

So you define some ordering on your pixels. The easiest and most obvious is top-left to bottom-right 0 to 783.

Then whenver you generate, you can never look at anything that comes after you (including yourself).

How you sample from these is pretty interesting and worth understanding the details of.
But basically you just need to generate the samples one at a time, as you would suspect.
So you initialize a zero image. Then get the top pixel. Then you have to feed that through the network
again to get the next pixel and so on. You only set the pixel you are currently working on.
And since the network is masked, pixel24 never gets freaked out that the rest of the pixels are 0.
Even though this image is OOD, pixel24 doesn't see this. As long as pixel0-23 do their jobs, things will
work out.


If these ever seem confusing, just remember that the goal is to make sure you can never look at
something that is generated after you. All design choices are to enforce that and to help it train better.
If you have these constraints, then you can think how you would have designed it in the first place.

There is some quote about this guy at a EE conference who showed a circuit design. Nobody could tell what it did.
But then if the crowd was asked to design something that solved the same problem, they basically came up with the same choices.

Who said this.


## MADE
specific autoencoder idea

basically you sample some mask such that you assign the value for each neuron.
then you make sure that neuron k can only see values neuron k-1. 
Like all the non-zero edge inputs must be <k

So you can sample those. And then you do masks so that anything else is zero-ed out.

Bigger networks work better since you can have more connections between pixels.
But I found these are a pretty shitty method.
They are very specky in the output.
You are only seeing a fixed random subset of the previous neurons. And the wiring is
weird. It's interesting that you can do this type of thing, and how in fact it is an autoregressive
model. But I think it's a pretty bad idea.

Not sure how much work builds off this directly. But maybe good to know.

## Wavenet

I'm not sure I'd even call this one masking, since you don't really mask out any weights.
You just rely on the structure of the Wavenet dilated convolutions and you add a few zero
paddings in exactly the right spots so that you get the behavior that you want, where you can't see yourself or the future.


Basically you are going to use this wavenet idea.
And then treat mnist as a 1d sequence by flattening, as we have been doing.

// todo: wavenet image

Dilated convolutions vs. strided convolutions. Probably just borrow that guys picture.


This is specifically how you do it:
// todo: my hand drawn image

The first layer can't see itself, so you need the 2 pad.
And then you can gradually step up the receptive field with dilation.


Add image of the block. For extra complexity.
https://arxiv.org/pdf/1609.03499.pdf

## PixelCNN

This is basically the exact same idea as in Wavenet, but just the masking is different.



## Transformer

When you walk through all these other ideas, ImageGPT seems pretty obvious.
(This is maybe an example of where knowing the details helps you predict the future better because you can come up with the ideas that people actually do.)

Like why just apply the same idea of Wavenet to this?
Just autoregressively mask out different parts of the transformer and you can generate an image.
You will still need to run step-by-step forward passes to generate.

One difference is that you don't mask out the weights, you mask out the attention.
So in the softmax, you just set those elements to -inf.

