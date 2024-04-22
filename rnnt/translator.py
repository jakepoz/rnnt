# Module which will convert from pre-encoded audio features to 
# the input embedding of a neural network
import torch

from typing import List, Optional, Tuple

from rnnt.causalconv import CausalConv1d


class ConvTranslator(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        stride: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.conv1 = CausalConv1d(input_dim, hidden_dim, kernel_size=5, stride=stride, dilation=1)
        self.conv2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, dilation=1)

        self.linear = torch.nn.Linear(hidden_dim, output_dim)
        self.output_layer_norm = torch.nn.LayerNorm(output_dim)

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, input):
        x = input

        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)

        x = self.linear(x)
        x = self.output_layer_norm(x)

        return x