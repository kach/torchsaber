# [torchsaber: Elegant Dimensions for a more Civilized Age](https://github.com/kach/torchsaber)

> It is a period of civil war. Rebel spaceships have won their first victory
> against the awesome `RuntimeError` ([experimental support for Named Tensors
> in
> PyTorch](https://pytorch.org/tutorials/intermediate/named_tensor_tutorial.html)).

## Motivation

How often have you written or seen code like this?

```python
images = torch.randn(10, 3, 64, 64)
flipped = images.transpose(2, 3)
cropped = flipped[:, :, x:x+32, y:y+32]
flatten = cropped.view(-1, 32 * 32)
gray = flatten.sum(dim=2) / 3.
```

or gotten a mysterious error like

```python
RuntimeError: The size of tensor x (2) must match the size of tensor y (4) at non-singleton dimension 2
```

I don't care what universe you're from, that's got to hurt your eyes. What do
all those numbers mean? What order are the dimensions in by the end? Is
`dim=2` the right summation to average the channels? Who knows? Unnamed
dimensions lead to anger; anger leads to bugs; bugs lead to suffering.

**But, a ray of hope!** PyTorch has
[released](https://pytorch.org/tutorials/intermediate/named_tensor_tutorial.html)
experimental support for _named tensors:_ that is, tensors whose dimensions
have names rather than simply numeric indices. It's wonderful news, but it
still has some rough edges. For example, you have to refer to names using
hardcoded string literals everywhere; a typo (e.g. `'hieght'`) can break your
code in unexpected ways that aren't caught at runtime.

`torchsaber` is a **minimal syntactic sugar for named tensors**. Its goal is to
give you power (unlimited power!) by allowing all manipulation of dimensions to
be done _by name_ rather than by numeric indices or hardcoded string literals.
Dimensions and their manipulations are
**[first-class](https://en.wikipedia.org/wiki/First-class_citizen) objects**
that interface cleanly with PyTorch's user-facing API. For example, the above
code snippet becomes:

```python
from torchsaber import dims
batch, channel, height, width, features =
	dims('batch', 'channel', 'height', 'width', 'features')

images = torch.randn | batch(10) + channel(3) + height(64) + width(64)
flipped = images.permute(~width, ~height)
cropped = flipped | height[:32] + width[:32]
flatten = cropped.flatten([~height, ~width], ~features)
gray = flatten.sum(dim=~channel) / 3.
```

By "minimal" I mean the entire implementation is around 100 lines of code.

## Can I use it?

Sure! The easiest way to learn is to read the big comment at the top of
`torchsaber.py`. It's a literate doctest! Then `pip install torch torchvision
torchsaber` and `from torchsaber import dims` and enjoy. torchsaber tries to be
compatible with [the named tensor
docs](https://pytorch.org/docs/stable/named_tensor.html#named-tensors) and
should work with operators
[supported](https://pytorch.org/docs/stable/name_inference.html#named-tensors-operator-coverage)
by named tensors.

However, because named tensors are experimental, so is torchsaber. The _real_
goal of the project is to provoke some discussion around human-friendly designs
for the tensor programs of the future.

## (Some!) references

Here's a non-exhaustive list of prior work (many of these have their own
bibliographies you can follow)…

- [Tensors Considered Harmful](http://nlp.seas.harvard.edu/NamedTensor) and [Part 2](http://nlp.seas.harvard.edu/NamedTensor2), which led to named tensor support in PyTorch.
- [An older PyTorch named tensor proposal](https://github.com/pytorch/pytorch/issues/4164)
- [_Typesafe Abstractions for Tensor Operations_ (SCALA 2017)](https://arxiv.org/pdf/1710.06892.pdf)
- Other projects with similar goals:
  - [datarray](https://github.com/BIDS/datarray)
  - [xarray](http://xarray.pydata.org/en/stable/)
  - [tsalib](https://github.com/ofnote/tsalib)

## _More_ to say, have you?

Open an issue, file a PR, send me an email!
