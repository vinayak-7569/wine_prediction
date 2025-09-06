from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier
import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names, model_name='Model', is_classification=True):
    '''Evaluate the model and print performance metrics with SHAP and cross-validation.'''
    print(f"\n=== Evaluating {model_name.upper()} ===\n")

    if not is_classification:
        raise ValueError("This evaluation module supports only classification tasks.")

    # Predict on test set
    y_pred = model.predict(X_test)

    # Classification metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {acc:.4f} ({acc*100:.2f}%)")

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("Actual\\Predicted\tLow\tMed\tHigh")
    for i, row in enumerate(cm):
        print(f"{['Low', 'Med', 'High'][i]}\t\t" + "\t".join(str(x) for x in row))

    # Cross-validation score
    print("\nPerforming 5-fold cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # SHAP feature importance
    print("\nCalculating SHAP feature importance...")
    try:
        # Use TreeExplainer for tree-based models (RF, XGBoost, CatBoost) or ensemble's RF
        if isinstance(model, VotingClassifier):
            # Use first estimator (RF) for SHAP in ensemble
            explainer = shap.TreeExplainer(model.estimators_[0])
        else:
            explainer = shap.TreeExplainer(model)
        
        shap_values = explainer.shap_values(X_test)
        shap_importance = np.abs(shap_values[1]).mean(axis=0)  # Use class 1 (Medium) for importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': shap_importance
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 Important Features (SHAP) for {model_name}:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:25} {row['importance']:.4f}")
    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")

    return acc

def evaluate_multiple_models(models, X_train, X_test, y_train, y_test, feature_names):
    '''Evaluate multiple models and compare their performance.'''
    print("\n=== Comparing Multiple Models ===\n")
    results = {}
    for name, model in models.items():
        acc = evaluate_model(model, X_train, X_test, y_train, y_test, feature_names, model_name=name)
        results[name] = acc
    
    # Print comparison
    print("\n=== Model Comparison Summary ===")
    for name, acc in results.items():
        print(f"{name.upper()}: {acc:.4f} ({acc*100:.2f}%)")
    
    best_model = max(results, key=results.get)
    print(f"\nBest Model: {best_model.upper()} with Accuracy: {results[best_model]:.4f}")