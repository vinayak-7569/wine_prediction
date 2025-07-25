 Wine Quality Prediction Project

## Project Overview
This is a comprehensive machine learning project designed to classify red wine quality using physicochemical properties. The objective is to predict wine quality levels categorized as Low, Medium, and High based on features like alcohol, acidity, sulphates, and more. The final model, built using a RandomForestClassifier, achieved a validation accuracy of **86.88%**. The pipeline includes data preprocessing, feature engineering, model training, evaluation, and saving the model for deployment.

## Key Achievements
- **86.88% Accuracy**: Achieved using RandomForestClassifier with engineered features
- **Feature Engineering**: Custom features such as alcohol_sulphates and acid_ratio improved performance
- **Production Ready**: Trained model and reusable prediction script are saved
- **Clean Pipeline**: Fully modular and maintainable ML architecture

## Project Structure
wine_prediction/
├── config/
│ └── config.yaml # Configuration settings
├── data/
│ └── wine_quality.csv # Raw dataset
├── models/ # Trained model files
│ ├── production_wine_model.joblib
│ ├── feature_names.joblib
│ └── wine_predictor.py
├── outputs/ # Visualizations and evaluation reports
├── src/
│ ├── data_loader.py # Data loading functions
│ ├── preprocessing.py # Preprocessing & feature engineering
│ ├── production_model.py # Model training and saving
│ ├── evaluation.py # Accuracy and confusion matrix
│ ├── visualization.py # Feature importance plots
├── main.py # Entry point to run the full pipeline
├── requirements.txt # Project dependencies
└── README.md # This file

bash
Copy
Edit

## Quick Start

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/vinayak-7569/wine_prediction.git
cd wine_prediction

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

# Install required packages
pip install -r requirements.txt
2. Run the Main Pipeline
bash
Copy
Edit
python main.py
3. Expected Output
yaml
Copy
Edit
Starting Wine Quality Prediction Pipeline...
Accuracy: 0.8688 (86.88%)
Model saved: models/production_wine_model.joblib
Predictor code saved: models/wine_predictor.py
Model Performance
Model	Accuracy	Notes
Random Forest	86.88%	Final selected model (3 classes)
Baseline Model	~70.0%	Before feature engineering

Key Features
🔧 Feature Engineering
alcohol_sulphates: Interaction of alcohol and sulphates

acid_ratio: Balance of fixed and volatile acidity

alcohol_quality_proxy: Alcohol scaled with quality indicator

🎯 Classification Approach
Target variable grouped into three classes:

Low Quality: 3–4

Medium Quality: 5–6

High Quality: 7–8

Addressed imbalance using class weights

🚀 Production Artifacts
Trained model (production_wine_model.joblib)

Feature list (feature_names.joblib)

Inference script (wine_predictor.py)

Technical Stack
Python 3.13.1

pandas 2.2.3

numpy 1.26.4

scikit-learn 1.5.2

joblib 1.4.2

matplotlib / seaborn for visualization

Future Enhancements
SHAP-based model interpretability

Hyperparameter tuning (e.g., GridSearchCV)

Add XGBoost and CatBoost comparisons

Streamlit or Flask app for deployment

License
This project is licensed under the MIT License.

Contact
GitHub: @vinayak-7569

Project Link: Wine Quality Prediction
