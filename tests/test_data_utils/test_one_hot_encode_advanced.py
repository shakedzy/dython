import pytest
import numpy as np
import pandas as pd
from dython.data_utils import one_hot_encode


class TestOneHotEncodeAdvanced:
    """Advanced tests for one_hot_encode to improve coverage"""
    
    def test_one_hot_encode_with_classes_parameter(self):
        """Test one_hot_encode with explicit classes parameter"""
        lst = [0, 1, 2]
        # Specify more classes than exist in data
        result = one_hot_encode(lst, classes=5)
        
        assert result.shape == (3, 5)
        # Verify the encoding
        assert result[0, 0] == 1  # First element is 0
        assert result[1, 1] == 1  # Second element is 1
        assert result[2, 2] == 1  # Third element is 2
    
    def test_one_hot_encode_with_classes_exact(self):
        """Test one_hot_encode with exact number of classes"""
        lst = [0, 1, 2, 3]
        result = one_hot_encode(lst, classes=4)
        
        assert result.shape == (4, 4)
        # All diagonals should be 1
        for i in range(4):
            assert result[i, i] == 1
    
    def test_one_hot_encode_without_classes(self):
        """Test one_hot_encode without classes parameter (None)"""
        lst = [0, 1, 2]
        result = one_hot_encode(lst, classes=None)
        
        # Should automatically determine from max value (2 + 1 = 3 classes)
        assert result.shape == (3, 3)
    
    def test_one_hot_encode_with_pandas_series(self):
        """Test one_hot_encode with pandas Series input"""
        series = pd.Series([0, 1, 2, 0])
        result = one_hot_encode(series)
        
        assert result.shape == (4, 3)
        assert result[0, 0] == 1
        assert result[1, 1] == 1
        assert result[2, 2] == 1
        assert result[3, 0] == 1
    
    def test_one_hot_encode_with_numpy_array(self):
        """Test one_hot_encode with numpy array input"""
        arr = np.array([2, 1, 0, 2])
        result = one_hot_encode(arr)
        
        assert result.shape == (4, 3)
        assert result[0, 2] == 1
        assert result[1, 1] == 1
        assert result[2, 0] == 1
        assert result[3, 2] == 1
    
    def test_one_hot_encode_single_element(self):
        """Test one_hot_encode with single element"""
        lst = [0]
        result = one_hot_encode(lst)
        
        assert result.shape == (1, 1)
        assert result[0, 0] == 1
    
    def test_one_hot_encode_large_values(self):
        """Test one_hot_encode with large values"""
        lst = [0, 5, 10]
        result = one_hot_encode(lst)
        
        # Should create 11 classes (0 through 10)
        assert result.shape == (3, 11)
        assert result[0, 0] == 1
        assert result[1, 5] == 1
        assert result[2, 10] == 1
    
    def test_one_hot_encode_repeated_values(self):
        """Test one_hot_encode with repeated values"""
        lst = [1, 1, 1, 2, 2]
        result = one_hot_encode(lst)
        
        assert result.shape == (5, 3)
        # First three should encode to class 1
        assert result[0, 1] == 1
        assert result[1, 1] == 1
        assert result[2, 1] == 1
        # Last two should encode to class 2
        assert result[3, 2] == 1
        assert result[4, 2] == 1
    
    def test_one_hot_encode_all_zeros(self):
        """Test one_hot_encode with all zeros"""
        lst = [0, 0, 0, 0]
        result = one_hot_encode(lst)
        
        assert result.shape == (4, 1)
        # All should be encoded to class 0
        assert all(result[:, 0] == 1)
    
    def test_one_hot_encode_sequential(self):
        """Test one_hot_encode with sequential values"""
        lst = [0, 1, 2, 3, 4, 5]
        result = one_hot_encode(lst)
        
        assert result.shape == (6, 6)
        # Should be an identity matrix
        assert np.array_equal(result, np.eye(6))
    
    def test_one_hot_encode_with_float_that_converts_to_int(self):
        """Test one_hot_encode with floats that can be converted to int"""
        lst = [0.0, 1.0, 2.0]
        result = one_hot_encode(lst)
        
        assert result.shape == (3, 3)
        assert result[0, 0] == 1
        assert result[1, 1] == 1
        assert result[2, 2] == 1
    
    def test_one_hot_encode_output_dtype(self):
        """Test that output has correct dtype (float)"""
        lst = [0, 1, 2]
        result = one_hot_encode(lst)
        
        # Output should be float64
        assert result.dtype == np.float64
    
    def test_one_hot_encode_sum_per_row(self):
        """Test that each row sums to 1 (one-hot property)"""
        lst = [0, 1, 2, 3, 0, 2]
        result = one_hot_encode(lst)
        
        # Each row should sum to exactly 1
        row_sums = result.sum(axis=1)
        assert all(row_sums == 1)
    
    def test_one_hot_encode_classes_less_than_max(self):
        """Test one_hot_encode when classes is less than max value + 1"""
        lst = [0, 1, 2]
        # This might cause issues if classes < max+1, but let's test current behavior
        # If classes=2 but we have value 2, it should still work (or fail gracefully)
        try:
            result = one_hot_encode(lst, classes=2)
            # If it works, check the shape
            assert result.shape[1] == 2
        except (IndexError, ValueError):
            # It's also valid if it raises an error
            pass

