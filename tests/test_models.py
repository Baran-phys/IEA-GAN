"""
Testcases for the models.py module
"""
import json
import unittest

import torch

import model


class TestModelInit(unittest.TestCase):
    """
    Model init unittests
    """
    def setUp(self):
        """
        Load default config file
        """
        with open("./config.json", encoding="utf-8") as configfile:
            self.config = json.load(configfile)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_init(self):
        """
        Check that models are initialized successfully with the current config.json
        """
        model.Generator(**self.config).to(self.device)
        model.Discriminator(**self.config).to(self.device)


if __name__ == "__main__":
    unittest.main()
