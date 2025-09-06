# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


def preprocess_data(
    data: pd.DataFrame,
    target_column: str = "quality",
    is_classification: bool = True,
    k_features: int = 12,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Preprocess wine dataset for classification or regression.
    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    df = data.copy()

    # Handle missing values
    df = df.dropna()

    # -------------------------
    # Feature Engineering
    # -------------------------
    df["alcohol_sulphates"] = df["alcohol"] * df["sulphates"]
    df["acid_ratio"] = df["fixed acidity"] / (df["volatile acidity"] + 1)
    df["density_alcohol"] = df["density"] / (df["alcohol"] + 1)

    # -------------------------
    # Classification target
    # -------------------------
    if is_classification:
        def quality_to_class(q):
            if q <= 4:
                return 0  # Low
            elif 5 <= q <= 6:
                return 1  # Medium
            else:
                return 2  # High
        df[target_column] = df[target_column].apply(quality_to_class)

    # Features & target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    feature_names = X.columns

    # -------------------------
    # Scaling
    # -------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------
    # Train-Test Split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state,
        stratify=y if is_classification else None
    )

    # -------------------------
    # Outlier Removal (training only)
    # -------------------------
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    z_scores = np.abs(stats.zscore(X_train_df))
    outlier_mask = (z_scores < 3).all(axis=1)
    if np.sum(~outlier_mask) > 0:
        X_train = X_train[outlier_mask]
        y_train = pd.Series(y_train).iloc[outlier_mask].values

    # -------------------------
    # Feature Selection
    # -------------------------
    if k_features < X_train.shape[1]:
        selector = SelectKBest(score_func=f_classif, k=k_features)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        feature_names = feature_names[selector.get_support()]

    # -------------------------
    # SMOTE (training only, classification)
    # -------------------------
    if is_classification:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, feature_names, scaler


def group_quality_series(y_series):
    """
    Map numeric wine quality classes to labels.
    """
    return y_series.map({0: "Low", 1: "Medium", 2: "High"})
