import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import numpy as np
from sklearn.ensemble import VotingClassifier

def plot_feature_importance(model, X_test, feature_names, model_name='Model', save_dir='plots'):
    '''Plot SHAP-based feature importance for the trained model and save the plot.'''
    print(f"\n=== Plotting SHAP Feature Importance for {model_name.upper()} ===\n")

    # Ensure save directory exists
    project_root = os.path.dirname(os.path.dirname(__file__))
    save_dir_path = os.path.join(project_root, save_dir)
    os.makedirs(save_dir_path, exist_ok=True)

    # Convert X_test to DataFrame if it's a numpy array
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    # Calculate SHAP values
    try:
        # Use TreeExplainer for tree-based models or ensemble's RF component
        if isinstance(model, VotingClassifier):
            print("Using RandomForest component for SHAP in ensemble...")
            explainer = shap.TreeExplainer(model.estimators_[0])  # Use RF from ensemble
        else:
            explainer = shap.TreeExplainer(model)
        
        shap_values = explainer.shap_values(X_test)
        
        # Use SHAP values for class 1 (Medium) for importance
        importance = np.abs(shap_values[1]).mean(axis=0)
        feature_importance = pd.Series(importance, index=feature_names).sort_values(ascending=False)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importance, y=feature_importance.index, palette='viridis')
        plt.title(f'SHAP Feature Importance for {model_name}')
        plt.xlabel('Mean |SHAP Value|')
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(save_dir_path, f'shap_feature_importance_{model_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved: {plot_path}")

        # Display plot
        plt.show()

        return feature_importance

    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")
        print("Falling back to default feature importance...")
        
        # Fallback to model.feature_importances_ if available
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = pd.Series(importance, index=feature_names).sort_values(ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x=feature_importance, y=feature_importance.index, palette='viridis')
            plt.title(f'Feature Importance for {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(save_dir_path, f'feature_importance_{model_name.lower()}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved: {plot_path}")

            plt.show()
            return feature_importance
        else:
            print("No feature importance available.")
            return None