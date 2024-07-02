import sys

import torch
from torch._C import _add_docstr, _euler  # type: ignore[attr-defined]

# Note: This not only adds doc strings for functions in the linalg namespace, but
# also connects the torch.linalg Python namespace to the torch._C._linalg builtins.

cons2prim = _add_docstr(_euler.cons2prim, r"""
euler.cons2prim(input, out=None) -> Tensor

Converts the conserved variables to primitive variables.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example:
    >>> a = torch.randn(3, 5)
    >>> b = torch.euler.cons2prim(a)
""")
