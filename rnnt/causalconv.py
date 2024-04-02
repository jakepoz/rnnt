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
        self.padding = (kernel_size - 1) * dilation - stride + 1

        if additional_context < 0:
            raise ValueError("additional_context must be non-negative")
        
        if additional_context > self.padding:
            raise ValueError("additional_context can't be greater than the padding")
        
        self.additional_context = additional_context

        self.left_padding = self.padding - additional_context

    # Input shape is (N, C_in, L_in)
    def forward(self, x):
        # Right padding is always zero, because think about it: during training, you don't know what happens AFTER the training sample
        # Padding with zeros is not a valid assumption, so you just would need to shorten the output length by that amount
        x = torch.nn.functional.pad(x, (self.left_padding, 0))
        return self.conv(x)
    
    def streaming_forward(self, x, state):
        input = torch.cat((state, x), dim=2)
        assert input.shape[2] == (self.conv.kernel_size[0] - 1) * self.conv.dilation[0] + 1
        result = self.conv(input)

        # Update the state
        state = input[:, :, result.shape[2] * self.conv.stride[0]:]

        return result, state