"""Checking utils module."""
import os
import sys
import unittest
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

class TestUtilsFunctions(unittest.TestCase):

    def test_round64(self):
        self.assertEqual(round64(64), 64)
        self.assertEqual(round64(32), 64)
        self.assertEqual(round64(65), 128)

    def test_get_net(self):
        self.assertFalse(get_net().training)

    def test_create_circular_mask(self):
        err = create_circular_mask(2) - \
              np.array([[0, 0, 1, 0 ,0],
                        [0, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 0],
                        [0, 0, 1, 0 ,0]])
        self.assertAlmostEqual(np.abs(err).max(), 0)

    def test_preprocess(self):
        img = np.zeros((122, 128, 3))
        res, _, _ = preprocess(img)
        self.assertEqual(res.shape, (1, 3, 128, 128))

    def test_postprocess(self):
        img = np.zeros((1, 3, 122, 101))
        img = torch.Tensor(img)
        img[0][0][0][0] = 256
        img[0][0][0][1] = -1
        res = postprocess(img, 11, 100)
        self.assertEqual(res.shape, (11, 100, 3))
        self.assertLess(res.max(), 256)
        self.assertGreaterEqual(res.min(), 0)

if __name__ == '__main__':
    unittest.main()
