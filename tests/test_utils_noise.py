"""
Unittests for the IEA-GAN utils.noise module
"""
import unittest
import random

import torch

from utils import noise


class NoiseTest(unittest.TestCase):
    """
    Tests for utils.noise module
    """
    def test_uniform_noise(self):
        """
        Test uniform noise addition to tensor
        """
        scale = random.random()
        # Create uniform noise object with no inplace addition
        uniform_noise = noise.UniformNoise(scale=scale, inplace=False)
        # Test tensor
        data = torch.rand(10, 10)
        # Use manual seed to create noise data beforehand
        torch.manual_seed(42)
        noise_data = torch.rand_like(data)

        # Assert no inplace addition to data
        self.assertFalse(uniform_noise(data).equal(data))

        # Assert that output matches addition of data and noise
        torch.manual_seed(42)
        self.assertTrue(uniform_noise(data).equal(data+noise_data*scale))

        # inplace addition
        scale = random.random()
        uniform_noise = noise.UniformNoise(scale=scale, inplace=True)
        data = torch.rand(10, 10)
        copy_data = data.detach().clone()

        torch.manual_seed(42)
        noise_data = torch.rand_like(data)

        # Assert inplace addition to data
        torch.manual_seed(42)
        self.assertTrue(uniform_noise(data).equal(data))

        # Assert output
        self.assertTrue(data.equal(copy_data+scale*noise_data))


    def test_gaussian_noise(self):
        """
        Test gaussian noise addition to tensor
        """
        gaussian_noise = noise.GaussianNoise()
        data = torch.randn(10,10)
        self.assertEqual(data.shape, (10,10))
        self.assertTrue(gaussian_noise(data).equal(data))


if __name__ == "__main__":
    unittest.main()
