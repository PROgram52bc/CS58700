#
import torch
import numpy as onp
from typing import List, cast

import torch

def averaged(mat: torch.Tensor) -> torch.Tensor:
    """
    Average of all transformations.
    """
    buf = []
    
    # Loop for 4 rotations (0°, 90°, 180°, 270°)
    for d_rotate in range(4):
        # Loop for 4 flips (horizontal, vertical, both, none)
        for d_flip in range(4):
            # Apply rotation and flip
            rotated_mat = torch.rot90(mat, d_rotate, dims=(2, 3))  # Rotate the matrix
            flipped_mat = torch.flip(rotated_mat, [d_flip])         # Flip the matrix
            
            # Append the transformed matrix
            buf.append(flipped_mat)
    
    # Stack the transformed matrices
    stacked_mat = torch.stack(buf)
    
    # Take the mean across the batch (first axis)
    result = torch.mean(stacked_mat, dim=0)
    
    return result


class Model(torch.nn.Module):
    R"""
    Model.
    """
    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """

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

        buf_conv = []

        for (num_ins, num_outs) in zip(channels[:-1], channels[1:]):
            buf_conv.append(torch.nn.Conv2d(num_ins, num_outs,
                kernel_size=kernel_size_conv, stride=stride_size_conv,
                padding=PADDING))

        self.convs = torch.nn.ModuleList(buf_conv)
        self.pool = torch.nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_size_pool)

        # Create a list of Linear layers.
        # Number of layer neurons are given in `shapes` except for input.
        
        # compute the size of the input to the fully connected layer

        print("kernel_size_conv: {}".format(kernel_size_conv))
        print("kernel_size_pool: {}".format(kernel_size_pool))
        print("stride_size_conv: {}".format(stride_size_conv))
        print("stride_size_pool: {}".format(stride_size_pool))
        print("PADDING: {}".format(PADDING))

        height,width = size,size
        print("initial dimension: {} x {}".format(height, width))

        for i in range(len(channels) - 1):
            # for each convolution layer
            # size_out = (size_in - kernel_size + 2 * padding_size) // stride + 1
            height = (height - kernel_size_conv + 2 * PADDING) // stride_size_conv + 1
            width = (width - kernel_size_conv + 2 * PADDING) // stride_size_conv + 1
            print("{}th iteration: {} x {}".format(i, height, width))

            # for the pooling layer, no padding
            # size_out = (size_in - kernel_size + 2 * padding_size) // stride + 1
            height = (height - kernel_size_pool) // stride_size_pool + 1
            width = (width - kernel_size_pool) // stride_size_pool + 1
            print("{}th iteration with pooling: {} x {}".format(i, height, width))

        fcs_input = height * width * channels[-1]

        shapes = [fcs_input] + shapes
        buf = []
        for (num_ins, num_outs) in zip(shapes[:-1], shapes[1:]):
            #
            buf.append(torch.nn.Linear(num_ins, num_outs))
        self.linears = torch.nn.ModuleList(buf)

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
        for i,conv in enumerate(self.convs):
            x = self.pool(torch.nn.functional.relu(conv(x)))
        # flatten the result
        x = x.view(x.shape[0], -1) # an array of flattened result

        for linear in self.linears:
            x = torch.nn.functional.relu(linear(x))

        return x

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
        # YOU SHOULD FILL IN THIS FUNCTION

        proper_size = kernel_size_conv
        print("proper_size: {}".format(proper_size))

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

        print("self.basis.shape: {}".format(self.basis.shape))

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
        self.kernel_size_conv = kernel_size_conv
        self.kernel_size_pool = kernel_size_pool
        self.stride_size_conv = stride_size_conv
        self.stride_size_pool = stride_size_pool

        buf_weight = []
        buf_bias = []

        for (num_ins, num_outs) in zip(channels[:-1], channels[1:]):
            # weight : out * in * kernel * kernel
            # bias : out
            buf_weight.append(torch.nn.Parameter(torch.randn(num_outs, num_ins, self.basis.shape[0], 1), requires_grad=True))
            buf_bias.append(torch.nn.Parameter(torch.randn(num_outs),requires_grad=True))
            print("buf_weight.shape: {}".format(buf_weight[-1].shape))
            print("buf_bias.shape: {}".format(buf_bias[-1].shape))

        self.weights = torch.nn.ParameterList(buf_weight)
        self.biases = torch.nn.ParameterList(buf_bias)

        self.pool = torch.nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_size_pool)

        # Create a list of Linear layers.
        # Number of layer neurons are given in `shapes` except for input.
        
        # compute the size of the input to the fully connected layer

        print("kernel_size_conv: {}".format(kernel_size_conv))
        print("kernel_size_pool: {}".format(kernel_size_pool))
        print("stride_size_conv: {}".format(stride_size_conv))
        print("stride_size_pool: {}".format(stride_size_pool))
        print("PADDING: {}".format(PADDING))

        height,width = size,size
        print("initial dimension: {} x {}".format(height, width))

        for i in range(len(channels) - 1):
            # for each convolution layer
            # size_out = (size_in - kernel_size + 2 * padding_size) // stride + 1
            height = (height - kernel_size_conv + 2 * PADDING) // stride_size_conv + 1
            width = (width - kernel_size_conv + 2 * PADDING) // stride_size_conv + 1
            print("{}th iteration: {} x {}".format(i, height, width))

            # for the pooling layer,
            # size_out = (size_in - kernel_size + 2 * padding_size) // stride + 1
            height = (height - kernel_size_pool) // stride_size_pool + 1
            width = (width - kernel_size_pool) // stride_size_pool + 1
            print("{}th iteration with pooling: {} x {}".format(i, height, width))

        fcs_input = height * width * channels[-1]

        shapes = [fcs_input] + shapes
        buf = []
        for (num_ins, num_outs) in zip(shapes[:-1], shapes[1:]):
            #
            buf.append(torch.nn.Linear(num_ins, num_outs))
        self.linears = torch.nn.ModuleList(buf)


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

        # print("initial x: {}".format(x))

        for weight, bias in zip(self.weights, self.biases):
            # transform the weight matrix with the basis vectors
            weight = torch.mul(weight, self.basis) # "broadcast" the weights
            # print("self.basis.shape: {}".format(self.basis.shape))
            # print("weight.shape: {}".format(weight.shape))
            dim_out, dim_in, dim_basis, dim_kernsq = weight.shape
            weight = weight.sum(dim=-2)
            weight = weight.reshape(dim_out, dim_in, self.kernel_size_conv, self.kernel_size_conv)
            # pass through the conv layer
            x = torch.nn.functional.conv2d(x, weight=weight, bias=None, stride=self.stride_size_conv, padding=PADDING)
            # add bias uniformly to each output channel
            bias = bias.view(1, dim_out, 1, 1)
            x = x + bias
            # pass through pooling layer
            x = self.pool(torch.nn.functional.relu(x))

        # print("shape before averaging: {}".format(x.shape))
        x = averaged(x)
        # print("shape after averaging: {}".format(x.shape))
        # TODO: apply the transformations (e.g. rotate) on the feature maps and take an average
        # need to figure out the shape of the final x, should be (batch * out * kernel * kernel)

        # flatten the result
        # print("x after cnn: {}".format(x))
        x = x.view(x.shape[0], -1) # an array of flattened result

        for linear in self.linears:
            x = torch.nn.functional.relu(linear(x))
        # print("x after linear: {}".format(x.view(-1)[:10]))

        return x
