import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import pytest
from src.preprocessing import preprocess_data

def test_preprocess_data():
    # Create a sample DataFrame
    df = pd.DataFrame({
        'fixed acidity': [7.4, 7.8, 7.0],
        'volatile acidity': [0.7, 0.88, 0.76],
        'quality': [5, 6, 5],
        'Id': [1, 2, 3]
    })
    
    # Test preprocessing
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df, 'quality', is_classification=True)
    
    assert X_train.shape[0] > 0, 'X_train should have rows'
    assert X_test.shape[0] > 0, 'X_test should have rows'
    assert len(y_train) > 0, 'y_train should have values'
    assert len(y_test) > 0, 'y_test should have values'
    assert 'Id' not in feature_names, 'Id column should be dropped'
    assert 'fixed acidity' in feature_names, 'Feature names should include fixed_acidity'