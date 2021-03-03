# Autoregressive models

For autoregressive models, you need to ensure that you can only look at the past, both during training and during sampling.

There are broadly two ways of achieving this: RNNs and masking.

With RNNs, you just feed the data points one-at-a-time and this ensures you never see the future.
This is straighforward.

With masking, you generally use some type of feed-forward networks and you have to somehow mask out
part of the architecture so that you can't see yourself or the future when you are trying to train or sample.

There are various ways to do this, which each of these files use a different approach.
MADE is somewhat hard to grok, but the rest of the approaches make quite a bit of sense if you imagine
that your entire goal is to make sure that you can only look at the past.
If you thought hard enough about how you would set that up, you could have figured it out.
That is where these solutions come from.

As described in the Berkeley course lectures, generally you can categorize masks into two different types: A and B.
A-type mask means you set it up so you can only see your past (not yourself, so you can't cheat)
B-type mask means you set it up so you can only see yourself and your past.
As long as you mix this B-types with at least one A-type, the autoregressive property will hold.


<!--TODO: include my drawings and longer explanations -->
