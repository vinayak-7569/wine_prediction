import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import pytest
from src.data_loader import load_data

def test_load_data(tmp_path):
    # Create a temporary CSV file for testing
    df = pd.DataFrame({
        'fixed acidity': [7.4, 7.8, 7.0],
        'volatile acidity': [0.7, 0.88, 0.76],
        'quality': [5, 6, 5],
        'Id': [1, 2, 3]
    })
    file_path = tmp_path / 'wine_data.csv'
    df.to_csv(file_path, index=False)
    
    # Test loading data
    data = load_data(str(file_path))
    assert isinstance(data, pd.DataFrame), 'Loaded data should be a pandas DataFrame'
    assert data.shape == (3, 4), 'Data shape should match input'
    assert 'quality' in data.columns, 'Quality column should be present'