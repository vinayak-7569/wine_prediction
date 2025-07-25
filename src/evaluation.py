from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test, is_classification=True):
    '''Evaluate the model and print performance metrics.'''
    y_pred = model.predict(X_test)

    if is_classification:
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
    else:
        print("R² Score:", r2_score(y_test, y_pred))
        print("MSE:", mean_squared_error(y_test, y_pred))
