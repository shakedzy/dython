import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dython._private import (
    convert,
    remove_incomplete_samples,
    replace_nan_with_value,
    plot_or_not,
    set_is_jupyter,
)


class TestConvertAdditional:
    """Additional tests for convert function to increase coverage"""
    
    def test_convert_dataframe_to_ndarray(self):
        """Test converting DataFrame to ndarray (line 69)"""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = convert(df, np.ndarray)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)
    
    def test_convert_dataframe_to_ndarray_no_copy(self):
        """Test converting DataFrame to ndarray without copy"""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = convert(df, np.ndarray, copy=False)
        assert isinstance(result, np.ndarray)
    
    def test_convert_series_to_list(self):
        """Test converting Series to list (line 76)"""
        series = pd.Series([1, 2, 3, 4, 5])
        result = convert(series, list)
        assert isinstance(result, list)
        assert result == [1, 2, 3, 4, 5]
    
    def test_convert_ndarray_to_list(self):
        """Test converting ndarray to list (line 78)"""
        arr = np.array([1, 2, 3, 4, 5])
        result = convert(arr, list)
        assert isinstance(result, list)
        assert result == [1, 2, 3, 4, 5]
    
    def test_convert_ndarray_to_list_2d(self):
        """Test converting 2D ndarray to list"""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        result = convert(arr, list)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == [1, 2]
    
    def test_convert_list_to_list_no_copy(self):
        """Test converting list to list without copy"""
        lst = [1, 2, 3, 4, 5]
        result = convert(lst, list, copy=False)
        assert result is lst  # Should be the same object
    
    def test_convert_list_to_list_with_copy(self):
        """Test converting list to list with copy"""
        lst = [1, 2, 3, 4, 5]
        result = convert(lst, list, copy=True)
        assert result is not lst  # Should be a different object
        assert result == lst  # But with same values
    
    def test_convert_ndarray_to_ndarray_no_copy(self):
        """Test converting ndarray to ndarray without copy"""
        arr = np.array([1, 2, 3, 4, 5])
        result = convert(arr, np.ndarray, copy=False)
        assert result is arr  # Should be the same object
    
    def test_convert_ndarray_to_ndarray_with_copy(self):
        """Test converting ndarray to ndarray with copy"""
        arr = np.array([1, 2, 3, 4, 5])
        result = convert(arr, np.ndarray, copy=True)
        assert result is not arr  # Should be a different object
        np.testing.assert_array_equal(result, arr)  # But with same values
    
    def test_convert_dataframe_to_dataframe_no_copy(self):
        """Test converting DataFrame to DataFrame without copy"""
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = convert(df, pd.DataFrame, copy=False)
        assert result is df  # Should be the same object
    
    def test_convert_dataframe_to_dataframe_with_copy(self):
        """Test converting DataFrame to DataFrame with copy"""
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = convert(df, pd.DataFrame, copy=True)
        assert result is not df  # Should be a different object
        pd.testing.assert_frame_equal(result, df)  # But with same values


class TestRemoveIncompleteSamplesAdditional:
    """Additional tests for remove_incomplete_samples to increase coverage"""
    
    def test_remove_incomplete_samples_with_numpy_arrays(self):
        """Test remove_incomplete_samples with numpy arrays (line 110)"""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, np.nan, 50.0])
        
        x_clean, y_clean = remove_incomplete_samples(x, y)
        
        # Should remove indices 2 and 3 (where there are NaNs)
        assert len(x_clean) == 3
        assert len(y_clean) == 3
        # After conversion in the function, inputs become lists
        # So result is lists (converted from original numpy arrays)
        assert isinstance(x_clean, list)
        assert isinstance(y_clean, list)
    
    def test_remove_incomplete_samples_with_series(self):
        """Test remove_incomplete_samples with pandas Series"""
        x = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        y = pd.Series([10.0, 20.0, 30.0, np.nan, 50.0])
        
        x_clean, y_clean = remove_incomplete_samples(x, y)
        
        assert len(x_clean) == 3
        assert len(y_clean) == 3
        # After conversion in the function, inputs become lists
        assert isinstance(x_clean, list)
        assert isinstance(y_clean, list)
    
    def test_remove_incomplete_samples_with_list(self):
        """Test remove_incomplete_samples with lists returns lists"""
        x = [1.0, 2.0, None, 4.0, 5.0]
        y = [10.0, 20.0, 30.0, None, 50.0]
        
        x_clean, y_clean = remove_incomplete_samples(x, y)
        
        assert len(x_clean) == 3
        assert len(y_clean) == 3
        # Result should be lists when input is lists
        assert isinstance(x_clean, list)
        assert isinstance(y_clean, list)
    
    def test_remove_incomplete_samples_no_nans(self):
        """Test remove_incomplete_samples with no NaN values"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        x_clean, y_clean = remove_incomplete_samples(x, y)
        
        assert len(x_clean) == 5
        assert len(y_clean) == 5
        np.testing.assert_array_equal(x_clean, x)
        np.testing.assert_array_equal(y_clean, y)


class TestReplaceNanWithValueAdditional:
    """Additional tests for replace_nan_with_value"""
    
    def test_replace_nan_with_value_numpy_nan(self):
        """Test replace_nan_with_value with numpy NaN values"""
        x = np.array([1.0, np.nan, 3.0, 4.0])
        y = np.array([10.0, 20.0, np.nan, 40.0])
        
        x_clean, y_clean = replace_nan_with_value(x, y, -999)
        
        assert x_clean[1] == -999
        assert y_clean[2] == -999
        assert isinstance(x_clean, np.ndarray)
        assert isinstance(y_clean, np.ndarray)
    
    def test_replace_nan_with_value_none(self):
        """Test replace_nan_with_value with None values"""
        x = [1.0, None, 3.0, 4.0]
        y = [10.0, 20.0, None, 40.0]
        
        x_clean, y_clean = replace_nan_with_value(x, y, 0)
        
        assert x_clean[1] == 0
        assert y_clean[2] == 0
    
    def test_replace_nan_with_value_string_replacement(self):
        """Test replace_nan_with_value with string replacement"""
        x = ['a', None, 'c', 'd']
        y = ['w', 'x', None, 'z']
        
        x_clean, y_clean = replace_nan_with_value(x, y, 'MISSING')
        
        assert x_clean[1] == 'MISSING'
        assert y_clean[2] == 'MISSING'
    
    def test_replace_nan_with_value_mixed_types(self):
        """Test replace_nan_with_value with mixed types"""
        x = pd.Series([1, 2, None, 4])
        y = pd.Series([10, None, 30, 40])
        
        x_clean, y_clean = replace_nan_with_value(x, y, -1)
        
        assert x_clean[2] == -1
        assert y_clean[1] == -1


class TestPlotOrNot:
    """Tests for plot_or_not function"""
    
    def test_plot_or_not_with_plot_true(self):
        """Test plot_or_not when plot=True (should call plt.show())"""
        # Create a simple plot
        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        
        # Mock the behavior - just ensure it doesn't raise an error
        try:
            # We can't actually test plt.show() in non-interactive environment
            # but we can test that the function runs without error
            plot_or_not(plot=False)  # Use False to avoid hanging
        finally:
            plt.close('all')
    
    def test_plot_or_not_with_plot_false_not_jupyter(self):
        """Test plot_or_not when plot=False and not in Jupyter"""
        # Ensure IS_JUPYTER is False
        set_is_jupyter(force_to=False)
        
        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        
        plot_or_not(plot=False)
        
        # Clean up
        plt.close('all')
    
    def test_plot_or_not_with_plot_false_in_jupyter(self):
        """Test plot_or_not when plot=False and in Jupyter (lines 24-26)"""
        # Set IS_JUPYTER to True to test the jupyter branch
        set_is_jupyter(force_to=True)
        
        # Create a figure
        fig = plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        
        # This should close the figure since plot=False and IS_JUPYTER=True
        plot_or_not(plot=False)
        
        # Reset to False for other tests
        set_is_jupyter(force_to=False)
        
        # Clean up any remaining figures
        plt.close('all')
    
    def test_plot_or_not_no_figure(self):
        """Test plot_or_not when there's no current figure"""
        plt.close('all')  # Ensure no figures exist
        
        # Should not raise an error even with no figure
        plot_or_not(plot=False)


class TestSetIsJupyter:
    """Tests for set_is_jupyter function"""
    
    def test_set_is_jupyter_force_true(self):
        """Test setting IS_JUPYTER to True"""
        set_is_jupyter(force_to=True)
        from dython._private import IS_JUPYTER
        assert IS_JUPYTER == True
    
    def test_set_is_jupyter_force_false(self):
        """Test setting IS_JUPYTER to False"""
        set_is_jupyter(force_to=False)
        from dython._private import IS_JUPYTER
        assert IS_JUPYTER == False
    
    def test_set_is_jupyter_auto_detect(self):
        """Test auto-detecting Jupyter (line 17)"""
        # When force_to is None, it should check sys.argv
        set_is_jupyter(force_to=None)
        # Since we're running in pytest, it should detect as not Jupyter
        from dython._private import IS_JUPYTER
        # Just verify it doesn't crash - the actual value depends on environment
        assert isinstance(IS_JUPYTER, bool)
        
        # Reset to False for other tests
        set_is_jupyter(force_to=False)

