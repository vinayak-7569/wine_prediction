from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def create_production_model(model_type, X_train, X_test, y_train, y_test, already_grouped=False):
    """
    Creates, trains, and evaluates a classification model.
    - model_type: 'rf', 'xgb', 'cat', or 'ensemble'
    - already_grouped: set True if y_train/y_test are already grouped into 0/1/2
    """
    print(f"\n=== Creating {model_type.upper()} Model ===\n")

    # Group labels if not already grouped
    if not already_grouped:
        y_train_grouped = np.array([group_quality(q) for q in y_train])
        y_test_grouped = np.array([group_quality(q) for q in y_test])
    else:
        y_train_grouped = np.array(y_train)
        y_test_grouped = np.array(y_test)

    # Check that there are at least 2 classes
    if len(np.unique(y_train_grouped)) < 2:
        raise ValueError("❌ Training target has only one unique class. Cannot train model.")

    # Define base models
    models = {
        'rf': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'xgb': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        ),
        'cat': CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            verbose=0,
            random_state=42
        )
    }

    # Initialize requested model
    if model_type in models:
        model = models[model_type]
        print(f"Initialized {model_type.upper()} model.")
    elif model_type == 'ensemble':
        print("Creating ensemble VotingClassifier...")
        model = VotingClassifier(
            estimators=[
                ('rf', models['rf']),
                ('xgb', models['xgb']),
                ('cat', models['cat'])
            ],
            voting='soft'
        )
    else:
        raise ValueError("Invalid model_type. Choose from: 'rf', 'xgb', 'cat', or 'ensemble'.")

    # Train model
    model.fit(X_train, y_train_grouped)

    # Predict on test set
    preds = model.predict(X_test)
    acc = accuracy_score(y_test_grouped, preds)

    print(f"✅ {model_type.upper()} Accuracy: {acc:.4f}")
    return model, acc

def group_quality(q):
    """Groups wine quality score into 3 classes: 0 (low), 1 (medium), 2 (high)"""
    if q <= 4:
        return 0
    elif q <= 6:
        return 1
    else:
        return 2

def get_model_types():
    """Returns available model type keys"""
    return ['rf', 'xgb', 'cat', 'ensemble']
