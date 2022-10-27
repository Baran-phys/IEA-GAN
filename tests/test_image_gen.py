"""
Test image generation
"""
import json
import unittest

import torch

import model


class TestImageGeneration(unittest.TestCase):
    """
    Run image generation with Generator model
    """

    def setUp(self):
        # Set device to what is available
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        print(f"Run TestImageGeneration on {self.device}")
        # Open default config file
        with open("config.json", "r", encoding="utf8") as config_fp:
            self.config = json.load(config_fp)

    def test_image_generation(self):
        """
        Init Generator and run
        """
        # pylint: disable=not-callable
        generator_model = model.Model(self.config).to(self.device)
        data = model.generate(generator_model)
        self.assertTupleEqual(
            data.shape, (self.config["batch_size"], 250, 768)
        )


if __name__ == "__main__":
    unittest.main()
