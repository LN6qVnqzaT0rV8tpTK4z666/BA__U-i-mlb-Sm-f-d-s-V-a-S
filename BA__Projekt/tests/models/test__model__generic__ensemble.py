# BA__Projekt/tests/models/test__model__generic__ensemble.py

import unittest
import torch
from models.model__generic_ensemble import GenericEnsembleRegressor


class TestGenericEnsembleRegressor(unittest.TestCase):

    def setUp(self):
        """
        Setup any required configurations for the tests.
        """
        # Base configuration for the GenericRegressor models
        self.base_config = {
            'input_dim': 10,
            'output_dim': 1,
            # Add any necessary parameters for GenericRegressor initialization
        }
        
        # Create the ensemble regressor with 3 models and a fixed seed
        self.ensemble = GenericEnsembleRegressor(base_config=self.base_config, n_models=3, seed=42)

        # Generate some dummy data for testing (batch_size, input_dim)
        self.x = torch.randn(5, 10)  # batch_size=5, input_dim=10

    def test_initialization(self):
        """
        Test the initialization of the ensemble regressor.
        """
        # Check that the ensemble contains the correct number of models
        self.assertEqual(len(self.ensemble.models), 3)

    def test_forward(self):
        """
        Test the forward pass through the ensemble.
        """
        output = self.ensemble(self.x)

        # Ensure the output is a tensor (aggregated predictions)
        self.assertIsInstance(output, torch.Tensor)

        # The output should be of shape (batch_size, output_dim)
        self.assertEqual(output.shape, (5, 1))

    def test_forward_with_evidential_regression(self):
        """
        Test the forward pass when the models return evidential outputs.
        """
        # Modify the base config to simulate evidential regression output
        self.base_config['evidential'] = True
        
        # Create a new ensemble with the updated config
        ensemble_evidential = GenericEnsembleRegressor(base_config=self.base_config, n_models=3, seed=42)
        
        output = ensemble_evidential(self.x)
        
        # Ensure the output is a tuple (mu, v, alpha, beta)
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 4)

        # Check the shape of each output (mu, v, alpha, beta should all have shape (batch_size, output_dim))
        for item in output:
            self.assertEqual(item.shape, (5, 1))

    def test_get_individual_outputs(self):
        """
        Test the `get_individual_outputs` method to ensure individual model outputs are returned.
        """
        individual_outputs = self.ensemble.get_individual_outputs(self.x)

        # Ensure we get a list of outputs (one per model)
        self.assertIsInstance(individual_outputs, list)
        self.assertEqual(len(individual_outputs), 3)

        # Ensure each output is a tensor of shape (batch_size, output_dim)
        for output in individual_outputs:
            self.assertIsInstance(output, torch.Tensor)
            self.assertEqual(output.shape, (5, 1))

if __name__ == '__main__':
    unittest.main()

