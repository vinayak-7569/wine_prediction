import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import pytest
from src.preprocessing import preprocess_data

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

def test_preprocess_data(mock_data):
    """Test the preprocessing pipeline."""
    # Test preprocessing with default k_features
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(
        mock_data, target_column='quality', is_classification=True, k_features=5
    )

    # Basic assertions
    assert X_train.shape[0] > 0, 'X_train should have rows'
    assert X_test.shape[0] > 0, 'X_test should have rows'
    assert len(y_train) == X_train.shape[0], 'y_train should match X_train rows'
    assert len(y_test) == X_test.shape[0], 'y_test should match X_test rows'
    assert 'Id' not in feature_names, 'Id column should be dropped'
    assert 'fixed acidity' in feature_names, 'Feature names should include fixed_acidity'

    # Test feature engineering
    assert 'sugar_density_ratio' in feature_names, 'Feature engineering should include sugar_density_ratio'
    assert 'acidity_to_alcohol' in feature_names, 'Feature engineering should include acidity_to_alcohol'
    assert 'alcohol_sulphates' in feature_names, 'Feature engineering should include alcohol_sulphates'
    assert 'acid_ratio' in feature_names, 'Feature engineering should include acid_ratio'
    assert 'alcohol_quality_proxy' in feature_names, 'Feature engineering should include alcohol_quality_proxy'

    # Test feature selection
    assert len(feature_names) == 5, f'Expected 5 features from SelectKBest, got {len(feature_names)}'

    # Test train-test split ratio
    assert abs(X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]) - 0.8) < 0.05, 'Train-test split should be ~80/20'

    # Test stratified splitting
    train_class_dist = pd.Series(y_train).value_counts(normalize=True)
    test_class_dist = pd.Series(y_test).value_counts(normalize=True)
    for quality in [3, 4, 5, 6, 7, 8]:
        if quality in train_class_dist and quality in test_class_dist:
            assert abs(train_class_dist.get(quality, 0) - test_class_dist.get(quality, 0)) < 0.1, \
                f'Class {quality} distribution should be similar in train and test'

    # Test outlier removal
    # Since mock data has no extreme outliers, add a row with an outlier to test
    outlier_data = mock_data.copy()
    outlier_data.loc[0, 'fixed acidity'] = 100  # Extreme value
    X_train_outlier, _, y_train_outlier, _, _ = preprocess_data(
        outlier_data, target_column='quality', is_classification=True, k_features=5
    )
    assert X_train_outlier.shape[0] < len(outlier_data) * 0.8, 'Outlier removal should reduce training data size'

def test_preprocess_classification_only(mock_data):
    """Test that preprocessing raises an error for non-classification tasks."""
    with pytest.raises(ValueError, match="This preprocessing module supports only classification tasks"):
        preprocess_data(mock_data, target_column='quality', is_classification=False)

def test_preprocess_missing_values(mock_data):
    """Test handling of missing values."""
    data_with_missing = mock_data.copy()
    data_with_missing.loc[0, 'fixed acidity'] = np.nan
    initial_rows = len(data_with_missing)
    X_train, X_test, y_train, y_test, _ = preprocess_data(
        data_with_missing, target_column='quality', is_classification=True
    )
    assert X_train.shape[0] + X_test.shape[0] < initial_rows, 'Rows with missing values should be dropped'