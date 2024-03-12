import torch
import torch.nn.functional as F

from .causalconv import CausalConv1d

# Basic design from https://arxiv.org/pdf/1904.03288.pdf
class JasperBlock(torch.nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, dropout, num_sub_blocks, additional_context: int = 0):
        super(JasperBlock, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_sub_blocks = num_sub_blocks

        for i in range(num_sub_blocks):
            sub_in_channels = in_channels if i == 0 else out_channels
            self.convs.append(CausalConv1d(sub_in_channels, out_channels, kernel_size, 1, 1, additional_context=additional_context))
            self.bns.append(torch.nn.BatchNorm1d(out_channels))

        self.residual_conv = torch.nn.Conv1d(in_channels, out_channels, 1)
        self.residual_bn = torch.nn.BatchNorm1d(out_channels)

        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        input_x = x

        residual_x = self.residual_conv(input_x)
        residual_x = self.residual_bn(residual_x)

        for index, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x)
            x = bn(x)

            if index == self.num_sub_blocks - 1:
                x = x + residual_x

            # Using GELU instead of RELU here
            x = torch.nn.functional.gelu(x)
            x = self.dropout(x)

        return x


# Convolutional Audio Encoder, based on nvidia nemo Jasper
class AudioEncoder(torch.nn.Module):
    def __init__(self, input_features:int = 80, prologue_kernel_size:int = 11, prologue_stride: int=2, prologue_dilation: int=1,
                       blocks: list[JasperBlock]=[],
                       epilogue_features: int=896, epilogue_kernel_size: int=29, epilogue_stride: int=1, epilogue_dilation: int=2,
                       output_features: int=1024):
        super(AudioEncoder, self).__init__()

        self.blocks = torch.nn.Sequential()

        self.prologue_stride = prologue_stride
        self.epilogue_stride = epilogue_stride

        self.total_additional_context = 2 * 4 * 2 # 4 subblocks, 2 additional context each, but after a stride 2 downsampling

        # Prologue convolution to take 80-mel spectrograms, stride 2 to downsample
        first_block_input_size = blocks[0].in_channels

        self.blocks.append(CausalConv1d(input_features, first_block_input_size, prologue_kernel_size, prologue_stride, prologue_dilation))
        self.blocks.append(torch.nn.BatchNorm1d(first_block_input_size))
        self.blocks.append(torch.nn.GELU())

        # Main Jasper block section
        self.blocks.extend(blocks)

        # Epilogue with dilation 2 and large kernel size
        last_block_output_size = blocks[-1].out_channels

        self.blocks.append(CausalConv1d(last_block_output_size, epilogue_features, epilogue_kernel_size, epilogue_stride, epilogue_dilation))
        self.blocks.append(torch.nn.BatchNorm1d(epilogue_features))
        self.blocks.append(torch.nn.GELU())

        # Slightly different 1x1 convolution to get to the final features we will use later
        self.blocks.append(CausalConv1d(epilogue_features, output_features, 1, 1, 1))

    def forward(self, x):
        y = self.blocks(x)
        return y
    
    def calc_output_lens(self, input_lens):
        # ceiling of input_lens / stride
        return torch.ceil(input_lens / self.prologue_stride).int()
