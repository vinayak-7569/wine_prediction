import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data, target_column, is_classification=True):
    '''Preprocess the dataset: handle missing values, split features and target, scale features.'''
    # Handle missing values
    data = data.dropna()
    
    # Drop Id column if present
    data = data.drop(columns=['Id'], errors='ignore')
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns