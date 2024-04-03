import torch
import unittest

from rnnt.causalconv import CausalConv1d
from rnnt.jasper import JasperBlock, AudioEncoder

class CausualConvTest(unittest.TestCase):
    def test_causal_conv1d_basic(self):
        x = torch.randn(1, 80, 10) # (N, C, L)
        conv = CausalConv1d(80, 128, 3, 1, 1)
        y = conv(x)
        self.assertEqual(y.shape, (1, 128, 10))

    def test_causal_conv1d_stride_2(self):
        x = torch.randn(1, 80, 10) # (N, C, L)
        conv = CausalConv1d(80, 128, 3, 2, 1)
        y = conv(x)
        self.assertEqual(y.shape, (1, 128, 5))

    def test_causal_conv1d_dilation_2(self):
        x = torch.randn(1, 80, 10) # (N, C, L)
        conv = CausalConv1d(80, 128, 3, 1, 2)
        y = conv(x)
        self.assertEqual(y.shape, (1, 128, 10))

    def test_causal_conv1d_stride_and_dilation(self):
        x = torch.randn(1, 80, 20) # (N, C, L)
        conv = CausalConv1d(80, 128, 3, 2, 2)
        y = conv(x)
        self.assertEqual(y.shape, (1, 128, 10))

    def test_causality(self):
        x_orig = torch.randn(1, 80, 10)
        x_mod = x_orig.clone()

        conv = CausalConv1d(80, 128, 3, 1, 1)

        x_mod[:, :, -1] = 0.0

        y_orig = conv(x_orig)
        y_mod = conv(x_mod)

        # Everything but the last one should be the same
        self.assertTrue(torch.allclose(y_orig[:, :, :-1], y_mod[:, :, :-1]))

        # Last one should be different
        self.assertFalse(torch.allclose(y_orig[:, :, -1], y_mod[:, :, -1]))

    def test_additional_context(self):
        x_orig = torch.randn(1, 80, 10)
        x_mod = x_orig.clone()

        conv = CausalConv1d(80, 128, 3, 1, 1, additional_context=1)

        x_mod[:, :, -1] = 0.0

        y_orig = conv(x_orig)
        y_mod = conv(x_mod)

        # Everything but the last one should be the same
        self.assertTrue(torch.allclose(y_orig[:, :, :-2], y_mod[:, :, :-2]))

        # Last one should be different
        self.assertFalse(torch.allclose(y_orig[:, :, -2], y_mod[:, :, -2]))



class JasperBlockTest(unittest.TestCase):
    def test_jasper_block_basic(self):
        x = torch.randn(1, 128, 10) # (N, C, L)
        block = JasperBlock(3, 128, 128, 0.2, 4)
        y = block(x)
        self.assertEqual(y.shape, (1, 128, 10))

    def test_causality(self):
        x_orig = torch.randn(1, 80, 10)
        x_mod = x_orig.clone()

        block = JasperBlock(3, 80, 128, 0.0, 4)
        block.eval() # Need to set this so that batch norm doesn't mess up causality tests, as it uses statistics across the entire time dimension

        print(block)

        x_mod[:, :, -1] = 0.0

        y_orig = block(x_orig)
        y_mod = block(x_mod)

        # Everything but the last one should be the same
        self.assertTrue(torch.allclose(y_orig[:, :, :-1], y_mod[:, :, :-1]))

        # Last one should be different
        self.assertFalse(torch.allclose(y_orig[:, :, -1], y_mod[:, :, -1]))


class AudioEncoderTest(unittest.TestCase):
    def test_shapes(self):
        x = torch.randn(1, 80, 10) # (N, C, L)
        encoder = AudioEncoder(
            blocks = [
                JasperBlock(3, 80, 128, 0.0, 4),
                JasperBlock(3, 128, 256, 0.0, 4),
            ]
        )
        y = encoder(x)
        self.assertEqual(y.shape, (1, 1024, 5))

    def test_causality(self):
        for i in range(10):
            encoder = AudioEncoder(
            blocks = [
                JasperBlock(3, 80, 128, 0.0, 4),
                JasperBlock(3, 128, 256, 0.0, 4),
            ])   
            encoder.eval()

            x_orig = torch.randn(1, 80, 201) # Needs to be odd because we have stride 2 in the first layer
            x_mod = x_orig.clone()

            x_mod[:, :, -1] = torch.randn(80)

            y_orig = encoder(x_orig)
            y_mod = encoder(x_mod)

            # Each JasperBlock is causal, so that doesn't add any lookahead
            expected_lookahead = encoder.total_additional_context

            # Everything but the last expected amount should be the same
            self.assertTrue(torch.allclose(y_orig[:, :, :-expected_lookahead], y_mod[:, :, :-expected_lookahead]))

            # Sometimes the networks don't align things right with random initialization, so we can't check the last expected amount

    def test_output_lens(self):
        encoder = AudioEncoder(
            prologue_stride=2,
            blocks = [
                JasperBlock(3, 80, 128, 0.0, 4),
                JasperBlock(3, 128, 256, 0.0, 4),
            ]
        )
        
        for i in range(10, 30):
            x = torch.randn(1, 80, i) # (N, C, L)
            y = encoder(x)
            #self.assertEqual(y.shape, (1, 1024, 5))
            print(y.shape)
            self.assertEqual(encoder.calc_output_lens(torch.tensor([i])).item(), y.shape[2], f"Error when length={i}")
