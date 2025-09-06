import yaml
import os
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

from src.production_model import create_production_model
from src.data_loader import load_data
from src.preprocessing import preprocess_data, group_quality_series

def main():
    print("\n=== 🍷 Starting Wine Quality Prediction Pipeline ===\n")

    # CLI args
    parser = argparse.ArgumentParser(description="Wine Quality Prediction")
    parser.add_argument('--model', type=str, default='ensemble',
                        choices=['rf', 'xgb', 'cat', 'ensemble'],
                        help="Choose model: rf, xgb, cat, ensemble")
    args = parser.parse_args()

    # Load config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_path = os.path.join(current_dir, config['data_path'])
    target_column = config['target_column']
    is_classification = config.get('is_classification', True)

    # Load data
    print("📥 Loading data...")
    data = load_data(data_path)

    # Preprocess data
    print("🔄 Preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(
        data,
        target_column=target_column,
        is_classification=is_classification,
        k_features=12
    )

    print("Class distribution in y_train after preprocessing:\n", np.unique(y_train, return_counts=True))

    # Train and evaluate model
    if args.model != 'ensemble':
        print(f"\n🚀 Training {args.model.upper()} model...\n")
        model, acc = create_production_model(
            model_type=args.model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            already_grouped=True  # labels already 0/1/2 from preprocessing
        )
        print(f"{args.model.upper()} Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    else:
        print("\n🚀 Training ensemble model...\n")
        ensemble_model, ensemble_acc = create_production_model(
            model_type='ensemble',
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            already_grouped=True  # crucial to avoid single-class error
        )

        print("\n📊 Comparing with individual models...\n")
        models = {
            'RF': RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
            'XGB': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            'CAT': CatBoostClassifier(verbose=0, random_state=42)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"{name} Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

        print(f"\n🧠 Ensemble Test Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")

    print("\n✅ Pipeline complete!\n")

if __name__ == '__main__':
    main()
