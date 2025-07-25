import yaml
import os
import numpy as np
import pandas as pd
import joblib
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def create_production_model():
    """Create and save the final production-ready model."""
    print("\n=== CREATING PRODUCTION MODEL ===\n")

    # Load config
    project_root = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    data_path_key = 'data_path' if 'data_path' in config else 'ï»¿data_path'
    data_path = os.path.join(project_root, config[data_path_key])
    target_column = config['target_column']
    is_classification = config.get('is_classification', True)

    # Load and feature engineer
    data = load_data(data_path)

    def add_key_features(df):
        df = df.copy()
        df['alcohol_sulphates'] = df['alcohol'] * df['sulphates']
        df['acid_ratio'] = df['fixed acidity'] / (df['volatile acidity'] + 0.001)
        df['alcohol_quality_proxy'] = df['alcohol'] / (df['volatile acidity'] + 0.001)
        return df

    print("Engineering 3 key features...")
    data = add_key_features(data)

    # Preprocess
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(
        data, target_column, is_classification
    )

    # Group target into 3 classes
    def group_quality(q):
        if q <= 4: return 0
        elif q <= 6: return 1
        else: return 2

    y_train_grouped = np.array([group_quality(q) for q in y_train])
    y_test_grouped = np.array([group_quality(q) for q in y_test])

    print("Grouped wine qualities: Low (3-4), Medium (5-6), High (7-8)")

    # Train model
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train_grouped)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test_grouped, y_pred)
    print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_grouped, y_pred)
    print("Actual\\Predicted\tLow\tMed\tHigh")
    for i, row in enumerate(cm):
        print(f"{['Low','Med','High'][i]}\t\t" + "\t".join(str(x) for x in row))

    # Feature importance
    print("\nTop 10 Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:25} {row['importance']:.4f}")

    # Save model and features
    model_dir = os.path.join(project_root, 'models')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'production_wine_model.joblib')
    feature_path = os.path.join(model_dir, 'feature_names.joblib')
    joblib.dump(model, model_path)
    joblib.dump(feature_names, feature_path)

    print(f"\nModel saved: {model_path}")
    print(f"Features saved: {feature_path}")

    # Generate wine_predictor.py
    code_path = os.path.join(model_dir, 'wine_predictor.py')
    with open(code_path, 'w') as f:
        f.write(f"""
def predict_wine_quality(wine_features):
    import joblib
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    model = joblib.load('models/production_wine_model.joblib')
    feature_names = joblib.load('models/feature_names.joblib')

    features = dict(wine_features)
    features['alcohol_sulphates'] = features['alcohol'] * features['sulphates']
    features['acid_ratio'] = features['fixed acidity'] / (features['volatile acidity'] + 0.001)
    features['alcohol_quality_proxy'] = features['alcohol'] / (features['volatile acidity'] + 0.001)

    X = np.array([[features[name] for name in feature_names]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    prediction = model.predict(X_scaled)[0]
    probs = model.predict_proba(X_scaled)[0]
    return ['Low', 'Medium', 'High'][prediction], probs.tolist()
""")

    print(f"Predictor code saved: {code_path}")
    print("\n✅ Production Model Pipeline Complete!\n")
    return model, acc


# Run directly for testing
if __name__ == '__main__':
    create_production_model()
