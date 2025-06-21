# BA__Projekt/tests/models/test__model__generic.py

import unittest
import torch
from models.model__generic import GenericRegressor


class TestGenericRegressor(unittest.TestCase):

    def setUp(self):
        """
        Setup any required configurations for the tests.
        """
        # Configurations for a standard regression model
        self.base_config_regression = {
            'input_dim': 10,
            'hidden_dims': [64, 64],
            'output_type': 'regression',
            'output_dim': 1,
            'use_dropout': False,
            'dropout_p': 0.2,
            'flatten_input': False,
            'use_batchnorm': False,
            'activation_name': 'relu'
        }

        # Configurations for evidential regression model
        self.base_config_evidential = {
            'input_dim': 10,
            'hidden_dims': [64, 64],
            'output_type': 'evidential',
            'output_dim': 1,
            'use_dropout': False,
            'dropout_p': 0.2,
            'flatten_input': False,
            'use_batchnorm': False,
            'activation_name': 'relu'
        }

        # Create a standard regression model
        self.regressor = GenericRegressor(**self.base_config_regression)

        # Create an evidential regression model
        self.evidential_regressor = GenericRegressor(**self.base_config_evidential)

        # Generate some dummy data for testing (batch_size, input_dim)
        self.x = torch.randn(5, 10)  # batch_size=5, input_dim=10

    def test_initialization(self):
        """
        Test the initialization of the GenericRegressor models.
        """
        # Test standard regression model
        self.assertEqual(self.regressor.output_type, 'regression')
        self.assertEqual(self.regressor.output_dim, 1)
        self.assertTrue(hasattr(self.regressor, 'output'))

        # Test evidential regression model
        self.assertEqual(self.evidential_regressor.output_type, 'evidential')
        self.assertEqual(self.evidential_regressor.output_dim, 1)
        self.assertTrue(hasattr(self.evidential_regressor, 'out_mu'))
        self.assertTrue(hasattr(self.evidential_regressor, 'out_log_v'))
        self.assertTrue(hasattr(self.evidential_regressor, 'out_log_alpha'))
        self.assertTrue(hasattr(self.evidential_regressor, 'out_log_beta'))

    def test_forward_regression(self):
        """
        Test the forward pass for the standard regression model.
        """
        output = self.regressor(self.x)

        # Ensure the output is a tensor
        self.assertIsInstance(output, torch.Tensor)

        # The output should be of shape (batch_size, output_dim)
        self.assertEqual(output.shape, (5, 1))

    def test_forward_evidential(self):
        """
        Test the forward pass for the evidential regression model.
        """
        output = self.evidential_regressor(self.x)

        # Ensure the output is a tuple (mu, v, alpha, beta)
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 4)

        # Check the shape of each output (mu, v, alpha, beta should all have shape (batch_size, output_dim))
        for item in output:
            self.assertEqual(item.shape, (5, 1))

    def test_activation_function(self):
        """
        Test the activation function used in the model.
        """
        # Check if the activation function is ReLU by default
        self.assertIsInstance(self.regressor.hidden[1], torch.nn.ReLU)

    def test_model_initialization(self):
        """
        Test model weight initialization (using He initialization).
        """
        # Check if weights of the first linear layer are initialized correctly (using He initialization)
        first_layer_weights = self.regressor.hidden[0].weight
        self.assertFalse(torch.all(first_layer_weights == 0))  # Check if they are not all zeros

    def test_dropout(self):
        """
        Test if dropout layers are correctly included in the model.
        """
        # Check if the dropout layer is included in the model for the regression model
        self.assertTrue(isinstance(self.regressor.hidden[4], torch.nn.Dropout))

if __name__ == '__main__':
    unittest.main()

