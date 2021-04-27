import os
import sys
import unittest
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformer_net import *

class TestUtilsFunctions(unittest.TestCase):

    def test_transformer(self):
        net = TransformerNet()
        net.eval()
        with torch.no_grad():
            res = net(torch.zeros(1, 3, 128, 64))
        self.assertEqual(res.shape, (1, 3, 128, 64))

    def test_conv_layer(self):
        net = ConvLayer(8, 32, 3, 2)
        net.eval()
        with torch.no_grad():
            res = net(torch.zeros(1, 8, 10, 12))
        self.assertEqual(res.shape, (1, 32, 5, 6))

        net = ConvLayer(8, 32, 3, 1)
        net.eval()
        with torch.no_grad():
            res = net(torch.zeros(1, 8, 10, 12))
        self.assertEqual(res.shape, (1, 32, 10, 12))

    def test_residual_block(self):
        net = ResidualBlock(8)
        net.eval()
        with torch.no_grad():
            res = net(torch.zeros(1, 8, 10, 12))
        self.assertEqual(res.shape, (1, 8, 10, 12))

    def test_upsample_conv_layer(self):
        net = UpsampleConvLayer(8, 32, 3, 1)
        net.eval()
        with torch.no_grad():
            res = net(torch.zeros(1, 8, 10, 12))
        self.assertEqual(res.shape, (1, 32, 20, 24))

        net = UpsampleConvLayer(8, 32, 3, 2)
        net.eval()
        with torch.no_grad():
            res = net(torch.zeros(1, 8, 10, 12))
        self.assertEqual(res.shape, (1, 32, 10, 12))


if __name__ == '__main__':
    unittest.main()