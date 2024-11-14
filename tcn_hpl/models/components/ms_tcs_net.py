import copy
from typing import Sequence

import einops
import torch
from torch import nn
import torch.nn.functional as F


class MultiStageModel(nn.Module):
    def __init__(
        self,
        fc_sequence_dims: Sequence[int],
        fc_sequence_dropout_p: float,
        num_stages: int,
        num_layers: int,
        num_f_maps: int,
        dim: int,
        num_classes: int,
    ):
        """Initialize a `MultiStageModel` module.

        :param fc_sequence_dims: Create N*2 linear layers with u-net-like skip
            connections connecting inputs and outputs of the same dimensions.
            If an empty sequence is provided, then no FC layers are created
        :param fc_sequence_dropout_p: P-value for drop-out layers utilized in
            the FC u-net block.
        :param num_stages: Number of State Model Layers.
        :param num_layers: Number of Layers within each State Model.
        :param num_f_maps: Feature size within the state model
        :param dim: Feature size between state models.
        :param num_classes: Number of output classes.
        """
        super(MultiStageModel, self).__init__()

        # One FC sequence that is applied to a single frame's feature vector,
        self.frame_fc = LinearSkipBlock([dim] + list(fc_sequence_dims), fc_sequence_dropout_p)

        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList(
            [
                copy.deepcopy(
                    SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)
                )
                for s in range(num_stages - 1)
            ]
        )

    def forward(self, x, mask):
        # x shape: [batch_size, feat_dim, window_size]
        # mask shape: [batch_size, window_size]

        # Shape [batch_size, window_size, feat_dim]
        re_x = einops.rearrange(x, "b d w -> b w d")
        re_x = self.frame_fc(re_x)
        # Bring it back to input shape [batch_size, feat_dim, window_size]
        x = einops.rearrange(re_x, "b w d -> b d w")

        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, None, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs


class LinearSkipBlock(nn.Module):

    def __init__(self, dims: Sequence[int], dropout_p):
        """
        Simple linear skip connection block.

        This happens to make use of GELU as the linear unit utilized.

        :param dims: A number of internal dimensions, creating N*2 linear
            layers connecting each dimensional shift.
        :param dropout_p: P-value for the drop-out layers utilized.
        """
        super().__init__()
        self.encode = nn.ModuleList([
            nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.GELU(), nn.Dropout(dropout_p))
            for i in range(len(dims) - 1)
        ])
        self.decode = nn.ModuleList([
            nn.Sequential(nn.Linear(dims[i], dims[i-1]), nn.GELU(), nn.Dropout(dropout_p))
            for i in range(len(dims) - 1, 0, -1)
        ])

    def forward(self, x):
        acts = []
        for layer in self.encode:
            acts.append(x)
            x = layer(x)
        for layer, a in zip(self.decode, acts[::-1]):
            x = layer(x) + a
        return x


class LinearResidual(nn.Module):
    """
    Sequence of fully connected layers with residual connections.

    There is a single skip connection from the input to the output in order to
    connect the two.

    The number of layer specified refers to the number of *internal* layers
    between the linear layers that transforms the interface dimension
    (input/output) and the layer dimensions.

    :param interface_dim: Dimension of input and output tensors.
    :param layer_dim: Dimension of internal sequential residual layers.
    :param n_layers: Number of internal layers. This should be 0 or greater.
    """

    def __init__(
        self,
        interface_dim: int,
        layer_dim: int = 512,
        n_layers: int = 5,
        dropout_p: float = 0.25,
    ):
        super().__init__()
        self.l_first = nn.Sequential(nn.Linear(interface_dim, layer_dim), nn.GELU(), nn.Dropout(dropout_p))
        self.l_inner = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_dim, layer_dim), nn.GELU(), nn.Dropout(dropout_p))
            for i in range(n_layers)
        ])
        self.l_last = nn.Sequential(nn.Linear(layer_dim, interface_dim), nn.GELU(), nn.Dropout(dropout_p))

    def forward(self, x):
        out = self.l_first(x)
        for layer in self.l_inner:
            out = layer(out) + out
        out = self.l_last(out) + x
        return out


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps))
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):

        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, None, :]

        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.LeakyReLU(0.2)
        self.norm = nn.BatchNorm1d(out_channels)
        # self.pool = nn.MaxPool1d(kernel_size=3, stride=1)

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.activation(out)
        # out = self.pool(out)
        out = self.norm(out)
        out = self.dropout(out)
        return (x + out) * mask[:, None, :]
