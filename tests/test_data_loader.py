import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import pytest
from src.data_loader import load_data

@pytest.fixture
def mock_data():
    """Create mock wine quality data for testing."""
    np.random.seed(42)
    data = pd.DataFrame({
        'fixed acidity': np.random.uniform(4, 15, 100),
        'volatile acidity': np.random.uniform(0.1, 1.5, 100),
        'alcohol': np.random.uniform(8, 15, 100),
        'sulphates': np.random.uniform(0.3, 2.0, 100),
        'residual sugar': np.random.uniform(0.5, 15, 100),
        'density': np.random.uniform(0.99, 1.0, 100),
        'quality': np.random.choice([3, 4, 5, 6, 7, 8], 100, p=[0.1, 0.1, 0.4, 0.3, 0.05, 0.05]),
        'Id': range(100)
    })
    return data

def test_load_data(mock_data, tmp_path):
    """Test the data loading pipeline."""
    file_path = tmp_path / 'wine_data.csv'
    mock_data.to_csv(file_path, index=False)

    # Test loading data
    data = load_data(str(file_path), target_column='quality')

    # Basic assertions
    assert isinstance(data, pd.DataFrame), 'Loaded data should be a pandas DataFrame'
    assert data.shape == (100, 8), f'Expected shape (100, 8), got {data.shape}'
    assert 'quality' in data.columns, 'Quality column should be present'
    assert 'Id' in data.columns, 'Id column should be present'
    assert 'grouped_quality' in data.columns, 'Grouped quality column should be added'

def test_load_data_missing_values(mock_data, tmp_path):
    """Test handling of missing values."""
    data_with_missing = mock_data.copy()
    data_with_missing.loc[0, 'fixed acidity'] = np.nan
    data_with_missing.loc[1, 'volatile acidity'] = np.nan
    file_path = tmp_path / 'wine_data_missing.csv'
    data_with_missing.to_csv(file_path, index=False)

    data = load_data(str(file_path), target_column='quality')
    assert data.shape[0] == 98, f'Expected 98 rows after dropping missing values, got {data.shape[0]}'

def test_load_data_duplicates(mock_data, tmp_path):
    """Test handling of duplicate rows."""
    data_with_duplicates = pd.concat([mock_data, mock_data.iloc[:5]], ignore_index=True)
    file_path = tmp_path / 'wine_data_duplicates.csv'
    data_with_duplicates.to_csv(file_path, index=False)

    data = load_data(str(file_path), target_column='quality')
    assert data.shape[0] == 100, f'Expected 100 rows after dropping duplicates, got {data.shape[0]}'

def test_load_data_outliers(mock_data, tmp_path):
    """Test outlier removal using Z-scores."""
    data_with_outliers = mock_data.copy()
    data_with_outliers.loc[0, 'fixed acidity'] = 100  # Extreme outlier
    file_path = tmp_path / 'wine_data_outliers.csv'
    data_with_outliers.to_csv(file_path, index=False)

    data = load_data(str(file_path), target_column='quality')
    assert data.shape[0] < 100, f'Expected fewer than 100 rows after outlier removal, got {data.shape[0]}'

def test_load_data_class_distribution(mock_data, tmp_path):
    """Test class distribution output."""
    file_path = tmp_path / 'wine_data.csv'
    mock_data.to_csv(file_path, index=False)

    data = load_data(str(file_path), target_column='quality')
    class_dist = data['grouped_quality'].value_counts().sort_index()
    assert len(class_dist) == 3, 'Expected 3 classes (Low, Medium, High)'
    assert class_dist.index.tolist() == [0, 1, 2], 'Classes should be 0 (Low), 1 (Medium), 2 (High)'

def test_load_data_invalid_file(tmp_path):
    """Test error handling for invalid file."""
    invalid_path = tmp_path / 'nonexistent.csv'
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_data(str(invalid_path), target_column='quality')