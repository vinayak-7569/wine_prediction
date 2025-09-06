from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def group_quality(q):
    if q <= 4:
        return 0
    elif q <= 6:
        return 1
    else:
        return 2

def create_production_model(model_type, X_train, X_test, y_train, y_test):
    y_train_grouped = np.array([group_quality(q) for q in y_train])
    y_test_grouped = np.array([group_quality(q) for q in y_test])

    print(f"\n=== Creating {model_type.upper()} Model ===\n")

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

    if model_type in models:
        model = models[model_type]
        print(f"Initialized {model_type.upper()} with tuned hyperparameters.")
    elif model_type == 'ensemble':
        print("Creating ensemble VotingClassifier with tuned models...")
        model = VotingClassifier(
            estimators=[
                ('rf', models['rf']),
                ('xgb', models['xgb']),
                ('cat', models['cat'])
            ],
            voting='soft'
        )
    else:
        raise ValueError("Model type must be 'rf', 'xgb', 'cat', or 'ensemble'.")

    model.fit(X_train, y_train_grouped)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test_grouped, preds)
    return model, acc

def get_model_types():
    return ['rf', 'xgb', 'cat', 'ensemble']
