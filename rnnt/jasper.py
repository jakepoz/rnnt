import torch
import torch.nn.functional as F

from typing import Literal
from .causalconv import CausalConv1d

# Basic design from https://arxiv.org/pdf/1904.03288.pdf

# A note regarding normalization
# BatchNorm works only okay, because the batch size is actually pretty low during training (ex. 4)
# Though the fact that the sequence length is long means that it's not that bad
# But performance was a bit better with instance norm
# However, you have to be careful, because in streaming mode, the instance norm is calculated over a much
# smaller sequence length, so we use track_running_stats=False to use a better estimate of the mean and variance
class JasperBlock(torch.nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, dropout, num_sub_blocks, norm_type: Literal["batch", "instance", "instance_affine"] = "batch", additional_context: int = 0):
        super(JasperBlock, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_sub_blocks = num_sub_blocks

        for i in range(num_sub_blocks):
            sub_in_channels = in_channels if i == 0 else out_channels
            self.convs.append(CausalConv1d(sub_in_channels, out_channels, kernel_size, 1, 1, additional_context=additional_context))

            if norm_type == "batch":
                self.norms.append(torch.nn.BatchNorm1d(out_channels))
            elif norm_type == "instance":
                self.norms.append(torch.nn.InstanceNorm1d(out_channels, track_running_stats=False))
            elif norm_type == "instance_affine":
                self.norms.append(torch.nn.InstanceNorm1d(out_channels, track_running_stats=False, affine=True))

        self.residual_conv = torch.nn.Conv1d(in_channels, out_channels, 1)

        if norm_type == "batch":
            self.residual_norm = torch.nn.BatchNorm1d(out_channels)
        elif norm_type == "instance":
            self.residual_norm = torch.nn.InstanceNorm1d(out_channels, track_running_stats=False)
        elif norm_type == "instance_affine":
            self.residual_norm = torch.nn.InstanceNorm1d(out_channels, track_running_stats=False, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        input_x = x

        residual_x = self.residual_conv(input_x)
        residual_x = self.residual_norm(residual_x)

        for index, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x)
            x = norm(x)

            if index == self.num_sub_blocks - 1:
                x = x + residual_x

            # Using GELU instead of RELU here
            x = torch.nn.functional.gelu(x)
            x = self.dropout(x)

        return x
    
    def streaming_forward(self, x: torch.Tensor, states: list[torch.Tensor]):
        input_x = x
        new_states = []

        residual_x = self.residual_conv(input_x)
        residual_x = self.residual_norm(residual_x)

        for index, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x, new_state = conv.streaming_forward(x, states[index])
            x = norm(x)

            if index == self.num_sub_blocks - 1:
                x = x + residual_x

            # Using GELU instead of RELU here
            x = torch.nn.functional.gelu(x)
            x = self.dropout(x)

            new_states.append(new_state)

        return x, new_states


# Convolutional Audio Encoder, based on nvidia nemo Jasper
class AudioEncoder(torch.nn.Module):
    def __init__(self, input_features:int = 80, prologue_kernel_size:int = 11, prologue_stride: int=2, prologue_dilation: int=1,
                       blocks: list[JasperBlock]=[],
                       epilogue_features: int=896, epilogue_kernel_size: int=29, epilogue_stride: int=1, epilogue_dilation: int=2,
                       output_features: int=1024,
                       norm_type: Literal["batch", "instance", "instance_affine"] = "batch"):
        super(AudioEncoder, self).__init__()

        self.blocks = torch.nn.Sequential()

        self.prologue_stride = prologue_stride
        self.epilogue_stride = epilogue_stride

        # Prologue convolution to take 80-mel spectrograms, stride 2 to downsample
        first_block_input_size = blocks[0].in_channels if len(blocks) > 0 else epilogue_features

        self.blocks.append(CausalConv1d(input_features, first_block_input_size, prologue_kernel_size, prologue_stride, prologue_dilation))

        if norm_type == "batch":
            self.blocks.append(torch.nn.BatchNorm1d(first_block_input_size))
        elif norm_type == "instance":
            self.blocks.append(torch.nn.InstanceNorm1d(first_block_input_size, track_running_stats=False))
        elif norm_type == "instance_affine":
            self.blocks.append(torch.nn.InstanceNorm1d(first_block_input_size, track_running_stats=False, affine=True))

        self.blocks.append(torch.nn.GELU())

        # Main Jasper block section
        self.blocks.extend(blocks)

        # Epilogue with dilation 2 and large kernel size
        last_block_output_size = blocks[-1].out_channels if len(blocks) > 0 else first_block_input_size

        self.blocks.append(CausalConv1d(last_block_output_size, epilogue_features, epilogue_kernel_size, epilogue_stride, epilogue_dilation))

        if norm_type == "batch":
            self.blocks.append(torch.nn.BatchNorm1d(epilogue_features))
        elif norm_type == "instance":
            self.blocks.append(torch.nn.InstanceNorm1d(epilogue_features, track_running_stats=False))
        elif norm_type == "instance_affine":
            self.blocks.append(torch.nn.InstanceNorm1d(epilogue_features, track_running_stats=False, affine=True))

        self.blocks.append(torch.nn.GELU())

        # Slightly different 1x1 convolution to get to the final features we will use later
        self.blocks.append(torch.nn.Conv1d(epilogue_features, output_features, kernel_size=1, stride=1, dilation=1))

    def forward(self, x):
        y = self.blocks(x)
        return y
    
    def streaming_forward(self, x: torch.Tensor, state: list[torch.Tensor]):
        state_index = 0
        return_state = []

        for module in self.blocks:
            if isinstance(module, CausalConv1d):
                x, new_state = module.streaming_forward(x, state[state_index])
                return_state.append(new_state)
                state_index += 1
            elif isinstance(module, JasperBlock):
                x, new_states = module.streaming_forward(x, state[state_index:state_index + module.num_sub_blocks])
                return_state.extend(new_states)
                state_index += module.num_sub_blocks
            else:
                x = module(x)

        return x, return_state
    
    def streaming_init_state(self, batch_size):
        state = []

        for module in self.blocks:
            if isinstance(module, CausalConv1d):
                state.append(torch.zeros(batch_size, module.conv.in_channels, (module.conv.kernel_size[0] - 1) * module.conv.dilation[0] - module.conv.stride[0] + 1))
            if isinstance(module, JasperBlock):
                for i in range(module.num_sub_blocks):
                    sub_in_channels = module.in_channels if i == 0 else module.out_channels
                    state.append(torch.zeros(batch_size, sub_in_channels, (module.convs[i].conv.kernel_size[0] - 1) * module.convs[i].conv.dilation[0] - module.convs[i].conv.stride[0] + 1))
        
        return state
    
    def calc_output_lens(self, input_lens):
        output_lens = input_lens.clone()
        for module in self.blocks:
            if isinstance(module, CausalConv1d):
                # Calculate output lengths considering left padding, stride, kernel size, and dilation
                padding = module.left_padding
                stride = module.conv.stride[0]
                kernel_size = module.conv.kernel_size[0]
                dilation = module.conv.dilation[0]
                output_lens = ((output_lens + padding - dilation * (kernel_size - 1) - 1) // stride + 1)

        return output_lens
