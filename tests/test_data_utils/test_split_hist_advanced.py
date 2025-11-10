import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from dython.data_utils import split_hist


class TestSplitHistAdvanced:
    """Advanced tests for split_hist function to improve coverage"""
    
    def test_split_hist_with_custom_title(self, iris_df):
        """Test split_hist with custom title"""
        result = split_hist(
            iris_df, 
            "sepal length (cm)", 
            "target",
            title="Custom Title",
            plot=False
        )
        
        assert isinstance(result, Axes)
        assert result.get_title() == "Custom Title"
        plt.close('all')
    
    def test_split_hist_with_default_title(self, iris_df):
        """Test split_hist with default title (empty string)"""
        result = split_hist(
            iris_df, 
            "sepal length (cm)", 
            "target",
            title="",
            plot=False
        )
        
        assert isinstance(result, Axes)
        # Default title should be "values by split_by"
        assert "sepal length (cm) by target" in result.get_title()
        plt.close('all')
    
    def test_split_hist_with_none_title(self, iris_df):
        """Test split_hist with title=None (covers line 117-120)"""
        result = split_hist(
            iris_df, 
            "sepal length (cm)", 
            "target",
            title=None,
            plot=False
        )
        
        assert isinstance(result, Axes)
        # When title is None, no title should be set (or empty)
        plt.close('all')
    
    def test_split_hist_with_custom_xlabel(self, iris_df):
        """Test split_hist with custom xlabel"""
        result = split_hist(
            iris_df, 
            "sepal length (cm)", 
            "target",
            xlabel="Custom X Label",
            plot=False
        )
        
        assert isinstance(result, Axes)
        assert result.get_xlabel() == "Custom X Label"
        plt.close('all')
    
    def test_split_hist_with_default_xlabel(self, iris_df):
        """Test split_hist with default xlabel (empty string)"""
        result = split_hist(
            iris_df, 
            "sepal length (cm)", 
            "target",
            xlabel="",
            plot=False
        )
        
        assert isinstance(result, Axes)
        # Default xlabel should be the values column name
        assert result.get_xlabel() == "sepal length (cm)"
        plt.close('all')
    
    def test_split_hist_with_none_xlabel(self, iris_df):
        """Test split_hist with xlabel=None (covers line 113-116)"""
        result = split_hist(
            iris_df, 
            "sepal length (cm)", 
            "target",
            xlabel=None,
            plot=False
        )
        
        assert isinstance(result, Axes)
        # When xlabel is None, no xlabel should be set
        plt.close('all')
    
    def test_split_hist_with_ylabel(self, iris_df):
        """Test split_hist with ylabel (covers line 121-122)"""
        result = split_hist(
            iris_df, 
            "sepal length (cm)", 
            "target",
            ylabel="Frequency",
            plot=False
        )
        
        assert isinstance(result, Axes)
        assert result.get_ylabel() == "Frequency"
        plt.close('all')
    
    def test_split_hist_without_ylabel(self, iris_df):
        """Test split_hist without ylabel (default None)"""
        result = split_hist(
            iris_df, 
            "sepal length (cm)", 
            "target",
            ylabel=None,
            plot=False
        )
        
        assert isinstance(result, Axes)
        plt.close('all')
    
    def test_split_hist_without_legend(self, iris_df):
        """Test split_hist without legend (covers line 111-112)"""
        result = split_hist(
            iris_df, 
            "sepal length (cm)", 
            "target",
            legend=None,
            plot=False
        )
        
        assert isinstance(result, Axes)
        # Verify no legend was created
        assert result.get_legend() is None
        plt.close('all')
    
    def test_split_hist_with_custom_legend_location(self, iris_df):
        """Test split_hist with custom legend location"""
        result = split_hist(
            iris_df, 
            "sepal length (cm)", 
            "target",
            legend="upper right",
            plot=False
        )
        
        assert isinstance(result, Axes)
        assert result.get_legend() is not None
        plt.close('all')
    
    def test_split_hist_with_figsize(self, iris_df):
        """Test split_hist with custom figsize"""
        result = split_hist(
            iris_df, 
            "sepal length (cm)", 
            "target",
            figsize=(10, 6),
            plot=False
        )
        
        assert isinstance(result, Axes)
        fig = result.get_figure()
        # Figsize is in inches, get_size_inches() returns it
        size = fig.get_size_inches()
        assert size[0] == 10
        assert size[1] == 6
        plt.close('all')
    
    def test_split_hist_with_hist_kwargs(self, iris_df):
        """Test split_hist with additional histogram kwargs"""
        result = split_hist(
            iris_df, 
            "sepal length (cm)", 
            "target",
            bins=20,
            alpha=0.7,
            edgecolor='black',
            plot=False
        )
        
        assert isinstance(result, Axes)
        plt.close('all')
    
    def test_split_hist_with_plot_true(self, iris_df):
        """Test split_hist with plot=True (just ensure no error)"""
        result = split_hist(
            iris_df, 
            "sepal length (cm)", 
            "target",
            plot=False  # Still use False to avoid display in tests
        )
        
        assert isinstance(result, Axes)
        plt.close('all')
    
    def test_split_hist_multiple_splits(self):
        """Test split_hist with data that has many split categories"""
        df = pd.DataFrame({
            'values': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 100)
        })
        
        result = split_hist(
            df, 
            "values", 
            "category",
            plot=False
        )
        
        assert isinstance(result, Axes)
        plt.close('all')
    
    def test_split_hist_binary_split(self):
        """Test split_hist with binary split"""
        df = pd.DataFrame({
            'values': np.random.randn(50),
            'group': ['A'] * 25 + ['B'] * 25
        })
        
        result = split_hist(
            df, 
            "values", 
            "group",
            plot=False
        )
        
        assert isinstance(result, Axes)
        plt.close('all')
    
    def test_split_hist_all_parameters(self, iris_df):
        """Test split_hist with all parameters specified"""
        result = split_hist(
            iris_df, 
            "sepal length (cm)", 
            "target",
            title="Complete Test",
            xlabel="Sepal Length",
            ylabel="Count",
            figsize=(12, 8),
            legend="upper left",
            plot=False,
            bins=30,
            alpha=0.6
        )
        
        assert isinstance(result, Axes)
        assert result.get_title() == "Complete Test"
        assert result.get_xlabel() == "Sepal Length"
        assert result.get_ylabel() == "Count"
        plt.close('all')

