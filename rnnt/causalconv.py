import torch
import torch.nn.functional as F

# This wraps a traditional Conv1d layer so that it can't "peek" into the future at all
# This is accomplished by padding the input with zeros on the left side
# The input shape is (N, C_in, L) and the output shape is (N, C_out, L) where L is the input length
#
# You can also allow it to "peek" into the future by setting additional_context to a positive integer
class CausalConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, additional_context: int = 0):
        super(CausalConv1d, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation)
        self.padding = (kernel_size - 1) * dilation

        if additional_context < 0:
            raise ValueError("additional_context must be non-negative")
        
        if additional_context > self.padding:
            raise ValueError("additional_context can't be greater than the padding")
        
        self.additional_context = additional_context

        self.left_padding = self.padding - additional_context
        self.right_padding = additional_context

    # Input shape is (N, C_in, L_in)
    def forward(self, x):
        x = torch.nn.functional.pad(x, (self.left_padding, self.right_padding))
        return self.conv(x)