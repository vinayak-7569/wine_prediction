import pandas as pd
import numpy as np
from scipy import stats

def load_data(file_path, target_column='quality'):
    """Load and preprocess the wine quality dataset with validation and outlier handling."""
    print("\n=== Loading Dataset ===\n")

    # Load data
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    # Print basic info
    print("Dataset Info:")
    print(data.info())
    print("\nFirst 5 rows:")
    print(data.head())

    # Data validation
    print("\nValidating Data...")
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.any():
        print("Missing Values Detected:")
        print(missing_values[missing_values > 0])
        # Drop rows with missing values (alternative: impute with median)
        data = data.dropna()
        print("Dropped rows with missing values.")
    
    # Check for duplicates
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows.")
        data = data.drop_duplicates()
        print("Dropped duplicate rows.")

    # Handle outliers using Z-scores
    print("\nHandling Outliers (Z-score > 3)...")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(data[numeric_cols]))
    outlier_mask = (z_scores < 3).all(axis=1)
    outliers = len(data) - sum(outlier_mask)
    if outliers > 0:
        print(f"Found {outliers} rows with outliers (Z-score > 3).")
        data = data[outlier_mask]
        print("Removed outliers.")
    
    # Class distribution (assuming 'quality' is the target)
    print("\nClass Distribution (before grouping):")
    print(data[target_column].value_counts().sort_index())
    
    # Group quality into 3 classes for consistency with pipeline
    def group_quality(q):
        if q <= 4: return 0  # Low
        elif q <= 6: return 1  # Medium
        else: return 2  # High
    
    data['grouped_quality'] = data[target_column].apply(group_quality)
    print("\nClass Distribution (after grouping into Low, Medium, High):")
    print(data['grouped_quality'].value_counts().sort_index().rename({0: 'Low', 1: 'Medium', 2: 'High'}))

    print("\n✅ Data Loading and Validation Complete!\n")
    return data