# BA__Projekt/tests/BA__Programmierung/util/test__singleton.py

import unittest
from unittest.mock import patch

from BA__Programmierung.util.singleton import Singleton


class SingletonTestClass(metaclass=Singleton):
    """
    A simple class to test the Singleton behavior.
    """
    def __init__(self, value):
        self.value = value


class TestSingleton(unittest.TestCase):
    
    def test_singleton_enforcement(self):
        """
        Test that only one instance of a class using the Singleton metaclass exists.
        """
        # Create two instances of SingletonTestClass with different values
        instance1 = SingletonTestClass("First")
        instance2 = SingletonTestClass("Second")

        # Ensure both instances are the same (Singleton behavior)
        self.assertIs(instance1, instance2, "Singleton pattern failed: different instances created.")

    def test_singleton_initialization(self):
        """
        Test that the Singleton class is initialized correctly.
        """
        # Instantiate the SingletonTestClass
        instance = SingletonTestClass("First")

        # Ensure the value is correctly set
        self.assertEqual(instance.value, "First", "Singleton instance was not correctly initialized.")

    def test_multiple_instantiation_attempts(self):
        """
        Test that subsequent instantiations return the same instance.
        """
        instance1 = SingletonTestClass("First")
        instance2 = SingletonTestClass("Second")

        # Since it's a Singleton, both instances should point to the same object
        self.assertIs(instance1, instance2, "Multiple instantiations returned different instances.")

    def test_singleton_instance_identity(self):
        """
        Test that the Singleton class maintains the same identity across multiple accesses.
        """
        instance1 = SingletonTestClass("First")
        instance2 = SingletonTestClass("Second")

        # The ID of both instances should be the same (Singleton behavior)
        self.assertEqual(id(instance1), id(instance2), "Instances have different IDs, Singleton failed.")

    @patch("BA__Projekt.util.singleton.Singleton.__call__")
    def test_singleton_call_method(self, mock_call):
        """
        Test that the __call__ method of the Singleton metaclass works correctly.
        """
        # Instantiate SingletonTestClass
        instance = SingletonTestClass("First")
        
        # Simulate the call method being triggered
        mock_call.return_value = instance
        mock_call()
        
        # Ensure that the call method has been called and the result is the expected instance
        mock_call.assert_called_once()
        self.assertEqual(instance.value, "First", "Singleton instance was not called correctly.")


if __name__ == "__main__":
    unittest.main()
