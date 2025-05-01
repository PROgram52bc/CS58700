#
import torch
import numpy as onp
from typing import List, cast


class Model(torch.nn.Module):
    R"""
    Model.
    """
    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        ...


class MLP(Model):
    R"""
    MLP.
    """
    def __init__(self, /, *, size: int, shapes: List[int]) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        #
        buf = []
        shapes = [size * size] + shapes
        for (num_ins, num_outs) in zip(shapes[:-1], shapes[1:]):
            #
            buf.append(torch.nn.Linear(num_ins, num_outs))
        self.linears = torch.nn.ModuleList(buf)

    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        #
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = onp.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R"""
        Forward.
        """
        #
        x = torch.flatten(x, start_dim=1)
        for (l, linear) in enumerate(self.linears):
            #
            x = linear.forward(x)
            if l < len(self.linears) - 1:
                #
                x = torch.nn.functional.relu(x)
        return x


#
PADDING = 3


class CNN(torch.nn.Module):
    R"""
    CNN.
    """
    def __init__(
        self,
        /,
        *,
        size: int, channels: List[int], shapes: List[int],
        kernel_size_conv: int, stride_size_conv: int, kernel_size_pool: int,
        stride_size_pool: int,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        # Create a list of Conv2D layers and shared max-pooling layer.
        # Input and output channles are given in `channels`.
        # ```
        # buf_conv = []
        # ...
        # self.convs = torch.nn.ModuleList(buf_conv)
        # self.pool = ...
        # ```
        # YOU SHOULD FILL IN THIS FUNCTION
        ...

        # Create a list of Linear layers.
        # Number of layer neurons are given in `shapes` except for input.
        # ```
        # buf = []
        # ...
        # self.linears = torch.nn.ModuleList(buf)
        # ```
        # YOU SHOULD FILL IN THIS FUNCTION
        ...

    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        #
        for conv in self.convs:
            #
            (ch_outs, ch_ins, h, w) = conv.weight.data.size()
            num_ins = ch_ins * h * w
            num_outs = ch_outs * h * w
            a = onp.sqrt(6 / (num_ins + num_outs))
            conv.weight.data.uniform_(-a, a, generator=rng)
            conv.bias.data.zero_()
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = onp.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        R"""
        Forward.
        """
        # CNN forwarding whose activation functions should all be relu.
        # YOU SHOULD FILL IN THIS FUNCTION
        ...


class CGCNN(Model):
    R"""
    CGCNN.
    """
    def __init__(
        self,
        /,
        *,
        size: int, channels: List[int], shapes: List[int],
        kernel_size_conv: int, stride_size_conv: int, kernel_size_pool: int,
        stride_size_pool: int,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        # This will load precomputed eigenvectors.
        # You only need to define the proper size.
        # proper_size = ...
        # YOU SHOULD FILL IN THIS FUNCTION
        ...

        #
        self.basis: torch.Tensor

        # Loaded eigenvectos are stored in `self.basis`
        with open("rf-{:d}.npy".format(proper_size), "rb") as file:
            #
            onp.load(file)
            eigenvectors = onp.load(file)
        self.register_buffer(
            "basis",
            torch.from_numpy(eigenvectors).to(torch.get_default_dtype()),
        )

        # Create G-invariant CNN like CNN, but is invariant to rotation and
        # flipping.
        # linear is the same as CNN.
        # You only need to create G-invariant Conv2D weights and biases.
        # ```
        # buf_weight = []
        # buf_bias = []
        # ...
        # self.weights = torch.nn.ParameterList(buf_weight)
        # self.biases = torch.nn.ParameterList(buf_bias)
        # ```
        # YOU SHOULD FILL IN THIS FUNCTION
        ...

    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        #
        for (weight, bias) in zip(self.weights, self.biases):
            #
            (_, ch_ins, b1, b2) = weight.data.size()
            a = 1 / onp.sqrt(ch_ins * b1 * b2)
            weight.data.uniform_(-a, a, generator=rng)
            bias.data.zero_()
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = onp.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        R"""
        Forward.
        """
        # CG-CNN forwarding whose activation functions should all be relu.
        # Pay attention that your forwarding should be invariant to rotation
        # and flipping.
        # Thus, if you rotate x by 90 degree (see structures.py), output of
        # this function should not change.
        # YOU SHOULD FILL IN THIS FUNCTION
        ...