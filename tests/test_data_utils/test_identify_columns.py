import pytest
import numpy as np
import pandas as pd
from dython.data_utils import identify_columns_by_type, identify_columns_with_na


class TestIdentifyColumnsByType:
    """Tests for identify_columns_by_type function"""
    
    def test_identify_int_columns(self):
        """Test identifying integer columns"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4],
            'float_col': [1.0, 2.0, 3.0, 4.0],
            'str_col': ['a', 'b', 'c', 'd']
        })
        
        result = identify_columns_by_type(df, include=['int64'])
        assert 'int_col' in result
        assert 'float_col' not in result
        assert 'str_col' not in result
    
    def test_identify_float_columns(self):
        """Test identifying float columns"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4],
            'float_col': [1.0, 2.0, 3.0, 4.0],
            'str_col': ['a', 'b', 'c', 'd']
        })
        
        result = identify_columns_by_type(df, include=['float64'])
        assert 'float_col' in result
        assert 'int_col' not in result
        assert 'str_col' not in result
    
    def test_identify_object_columns(self):
        """Test identifying object columns"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4],
            'float_col': [1.0, 2.0, 3.0, 4.0],
            'str_col': ['a', 'b', 'c', 'd']
        })
        
        result = identify_columns_by_type(df, include=['object'])
        assert 'str_col' in result
        assert 'int_col' not in result
        assert 'float_col' not in result
    
    def test_identify_multiple_types(self):
        """Test identifying multiple column types"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4],
            'float_col': [1.0, 2.0, 3.0, 4.0],
            'str_col': ['a', 'b', 'c', 'd']
        })
        
        result = identify_columns_by_type(df, include=['int64', 'float64'])
        assert 'int_col' in result
        assert 'float_col' in result
        assert 'str_col' not in result
    
    def test_identify_category_columns(self):
        """Test identifying categorical columns"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4],
            'cat_col': pd.Categorical(['a', 'b', 'c', 'a'])
        })
        
        result = identify_columns_by_type(df, include=['category'])
        assert 'cat_col' in result
        assert 'int_col' not in result
    
    def test_identify_with_numpy_array(self):
        """Test identify_columns_by_type with numpy array"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        
        result = identify_columns_by_type(arr, include=['int64'])
        # Numpy arrays converted to DataFrame have default numeric types
        assert isinstance(result, list)
    
    def test_identify_no_matching_columns(self):
        """Test when no columns match the requested type"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4],
            'float_col': [1.0, 2.0, 3.0, 4.0]
        })
        
        result = identify_columns_by_type(df, include=['object'])
        assert result == []
    
    def test_identify_all_columns_match(self):
        """Test when all columns match the requested type"""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4],
            'col2': [5, 6, 7, 8],
            'col3': [9, 10, 11, 12]
        })
        
        result = identify_columns_by_type(df, include=['int64'])
        assert len(result) == 3
        assert 'col1' in result
        assert 'col2' in result
        assert 'col3' in result


class TestIdentifyColumnsWithNA:
    """Tests for identify_columns_with_na function"""
    
    def test_identify_columns_with_na_basic(self):
        """Test basic identification of columns with NA values"""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [5, 6, 7, 8],
            'col3': [np.nan, np.nan, 3, 4]
        })
        
        result = identify_columns_with_na(df)
        
        # Should return DataFrame with column and na_count
        assert isinstance(result, pd.DataFrame)
        assert 'column' in result.columns
        assert 'na_count' in result.columns
        
        # col3 should be first (2 NAs), then col1 (1 NA)
        assert len(result) == 2
        assert result.iloc[0]['column'] == 'col3'
        assert result.iloc[0]['na_count'] == 2
        assert result.iloc[1]['column'] == 'col1'
        assert result.iloc[1]['na_count'] == 1
    
    def test_identify_columns_with_na_none_values(self):
        """Test with None values (which pandas treats as NA)"""
        df = pd.DataFrame({
            'col1': [1, 2, None, 4],
            'col2': [5, None, None, 8]
        })
        
        result = identify_columns_with_na(df)
        
        assert len(result) == 2
        # col2 should be first (2 NAs)
        assert result.iloc[0]['column'] == 'col2'
        assert result.iloc[0]['na_count'] == 2
    
    def test_identify_columns_with_na_no_na(self):
        """Test when no columns have NA values"""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4],
            'col2': [5, 6, 7, 8],
            'col3': [9, 10, 11, 12]
        })
        
        result = identify_columns_with_na(df)
        
        # Should return empty DataFrame
        assert len(result) == 0
        assert 'column' in result.columns
        assert 'na_count' in result.columns
    
    def test_identify_columns_with_na_all_na(self):
        """Test when all values in a column are NA"""
        df = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan, np.nan],
            'col2': [1, 2, 3, 4],
            'col3': [np.nan, 6, np.nan, 8]
        })
        
        result = identify_columns_with_na(df)
        
        assert len(result) == 2
        # col1 should be first (4 NAs), then col3 (2 NAs)
        assert result.iloc[0]['column'] == 'col1'
        assert result.iloc[0]['na_count'] == 4
        assert result.iloc[1]['column'] == 'col3'
        assert result.iloc[1]['na_count'] == 2
    
    def test_identify_columns_with_na_string_columns(self):
        """Test with string columns containing NA"""
        df = pd.DataFrame({
            'str_col': ['a', np.nan, 'c', 'd'],
            'int_col': [1, 2, 3, 4],
            'mixed_col': ['x', None, 'z', np.nan]
        })
        
        result = identify_columns_with_na(df)
        
        assert len(result) == 2
        # Both str_col and mixed_col have NAs
        columns_with_na = result['column'].tolist()
        assert 'str_col' in columns_with_na
        assert 'mixed_col' in columns_with_na
        assert 'int_col' not in columns_with_na
    
    def test_identify_columns_with_na_sorted_order(self):
        """Test that results are sorted by na_count in descending order"""
        df = pd.DataFrame({
            'col1': [1, np.nan, 3, 4],
            'col2': [np.nan, np.nan, np.nan, 8],
            'col3': [9, 10, np.nan, np.nan],
            'col4': [13, 14, 15, 16]
        })
        
        result = identify_columns_with_na(df)
        
        # Should be sorted: col2 (3), col3 (2), col1 (1)
        assert len(result) == 3
        na_counts = result['na_count'].tolist()
        # Verify descending order
        assert na_counts == sorted(na_counts, reverse=True)
        assert result.iloc[0]['column'] == 'col2'
        assert result.iloc[1]['column'] == 'col3'
        assert result.iloc[2]['column'] == 'col1'
    
    def test_identify_columns_with_na_from_numpy(self):
        """Test with numpy array input"""
        arr = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
        
        result = identify_columns_with_na(arr)
        
        # Should work with numpy arrays converted to DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0  # Should detect NA values
    
    def test_identify_columns_with_na_mixed_types(self):
        """Test with mixed data types"""
        df = pd.DataFrame({
            'int_col': [1, 2, np.nan, 4],
            'float_col': [1.5, np.nan, 3.5, 4.5],
            'str_col': ['a', 'b', 'c', 'd'],
            'bool_col': [True, False, np.nan, True]
        })
        
        result = identify_columns_with_na(df)
        
        # int_col, float_col, and bool_col should have NA
        assert len(result) == 3
        columns_with_na = result['column'].tolist()
        assert 'int_col' in columns_with_na
        assert 'float_col' in columns_with_na
        assert 'bool_col' in columns_with_na
        assert 'str_col' not in columns_with_na

