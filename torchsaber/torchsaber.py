'''
Every tensor program should name its dimensions up front. The `dims` function
lets you produce a set of named dimensions, which can be passed around as the
building blocks for gnarly tensor operations.

>>> batch, channel, height, width = dims('batch', 'channel', 'height', 'width')

From now on, you should never have to hardcode string constants. If needed, you
can recover a dimension's name using `str`, or (as a shorthand) the unary
bitwise "not" operator `~`:

>>> str(batch)
'batch'
>>> ~channel
'channel'

Dimensions are versatile tools that can be combined in many ways. For example,
by specifying an extent you can build a `shape`.

>>> channel(3)
channel(3)

You can combine `shape`s to produce an (ordered) specification of a tensor's
shape.

>>> width(5) + height(10)
width(5) + height(10)

`shape`s can be used as constructors by piping standard PyTorch factory
functions.

>>> t = torch.randn | batch(10) + channel(3) + height(128) + width(128)
>>> t.shape
torch.Size([10, 3, 128, 128])
>>> t.names
('batch', 'channel', 'height', 'width')
>>> (torch.zeros | width(5) + height(10)).sum()
tensor(0.)
>>> (torch.ones  | width(5) + height(10)).sum(~width).mean(~height)
tensor(5.)

(Notice the use of `~width` and `~height` in that last example!)

`shape`s can also be used to lift anonymous tensors to named tensors in a way
that checks dimension size explicitly.

>>> torch.tensor([[1, 2], [3, 4]]) | width(2) + height(2)
tensor([[1, 2],
        [3, 4]], names=('width', 'height'))
>>> torch.tensor([[1, 2], [3, 4]]) | width(4) + height(1)
Traceback (most recent call last):
...
TypeError: Expected shape [4, 1] but got [2, 2]

Finally, if needed you can convert a `shape` to a torch `Size` using
`.to_torch()`:

>>> channel(3).to_torch()
torch.Size([3])

Once you have a tensor, you can access its elements by using `slice`s. You can
build a `slice` by subscripting a dimension. Like `shape`s, `slice`s can be
composed with `+`, though with `slices` the operation is commutative.

>>> batch[0] + channel[:2] + height[5::2]
batch[0] + channel[:2] + height[5::2]

Just like with `shape`s, you can pipe tensors into `slice`s.

>>> blue = t | channel[2]
>>> blue.shape
torch.Size([10, 128, 128])
>>> even_batches = t | batch[::2]
>>> even_batches.shape
torch.Size([5, 3, 128, 128])
>>> cropped = t | height[:64] + width[-64:]
>>> cropped.shape
torch.Size([10, 3, 64, 64])

...and that's all! Your next steps should be to read the documentation at:
1. https://pytorch.org/tutorials/intermediate/named_tensor_tutorial.html
2. https://pytorch.org/docs/stable/named_tensor.html
3. https://pytorch.org/docs/stable/name_inference.html#named-tensors-operator-coverage
Then go off to bring peace and justice to the galaxy!

'''

import torch

# "But Master Yoda said I should be mindful of the future."
# "Not at the expense of the moment."
import warnings
with warnings.catch_warnings() as w:
    warnings.simplefilter('ignore')
    _ = torch.ones(1, names=('_',))


class shape:

    def __init__(self, dims=[], ns=[]):
        self.dims = dims
        self.ns = ns
        self.ellipsis = False

    def to_torch(self):
        return torch.Size(self.ns)

    def __add__(self, other):
        return shape(
            self.dims + other.dims,
            self.ns + other.ns
        )

    def __ror__(self, other):
        if other in [
            torch.zeros, torch.ones, torch.randn,
            torch.empty, torch.rand
        ]:
            return other(self.ns, names=self.dims)
        if type(other) == torch.Tensor:
            if list(other.shape) != self.ns:
                raise TypeError(
                    "Expected shape " + str(self.ns) +
                    " but got " + str(list(other.shape))
                )
            return other.refine_names(*self.dims)
        raise TypeError("Piping something unexpected to shape " + repr(self))

    def __repr__(self):
        return ' + '.join([
            '%s(%d)' % (name, size)
            for name, size in zip(self.dims, self.ns)
        ])

class subscript:
    def __init__(self, lookup):
        self.lookup = lookup

    def __add__(self, other):
        return subscript({**self.lookup, **other.lookup})

    def __ror__(self, other):
        subs = []
        for key in self.lookup:
            if key not in other.names:
                real = ' + '.join([
                    '%s(%d)' % (name, size)
                    for name, size in zip(other.names, other.shape)
                ])
                raise TypeError(
                    'Tried to slice non-existent dimension \'%s\' '
                    'in tensor with dimensions ' % key + real
                )

        for key in other.names:
            if key not in self.lookup:
                subs.append(slice(None, None, None))
            else:
                subs.append(self.lookup[key])
        return other[subs]

    def __repr__(self):
        def nstr(s):
            if s is None:
                return ''
            return str(s)

        def slice_to_string(s):
            if type(s) == int:
                return str(s)
            if s.step is None:
                return nstr(s.start) + ':' + nstr(s.stop)
            return nstr(s.start) + ':' + nstr(s.stop) + ':' + nstr(s.step)

        return ' + '.join([
            '%s[%s]' % (name, slice_to_string(size))
            for name, size in self.lookup.items()
        ])

class dim:
    def __init__(self, name):
        self.name = name

    def __getitem__(self, key):
        return subscript({self.name: key})

    def __call__(self, n=...):
        return shape([self.name], [n])

    def __repr__(self):
        return self.name

    def __invert__(self):
        return self.name

    def __str__(self):
        return self.name

def dims(*args):
    if len(args) == 1:
        return dim(args[0])
    return tuple(map(dim, args))
