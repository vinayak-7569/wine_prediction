# Wine Quality Prediction

A machine learning pipeline to predict wine quality based on physicochemical properties. This project uses **RandomForest, XGBoost, CatBoost, and an ensemble VotingClassifier** to achieve high accuracy, with preprocessing, feature engineering, and SMOTE for class balancing.

---

## 🏗 Project Structure

wine_prediction1/
│
├─ data/ # Dataset files (wine_quality.csv)
├─ src/ # Source code modules
│ ├─ data_loader.py
│ ├─ preprocessing.py
│ ├─ production_model.py
│ ├─ model_factory.py
│ ├─ evaluation.py
│ ├─ visualization.py
│ └─ init.py
├─ tests/ # Pytest unit tests
│ └─ test_production.py
├─ main.py # Pipeline entry point
├─ requirements.txt # Python dependencies
└─ README.md # Project documentation

yaml
Copy code

---

## ⚙ Features

- **Data Preprocessing**
  - Handling duplicates and outliers
  - Feature engineering: `density_alcohol`, `acid_ratio`, `alcohol_sulphates`
  - SMOTE for class balancing
  - Feature selection with `SelectKBest`

- **Models**
  - RandomForest, XGBoost, CatBoost
  - Ensemble VotingClassifier

- **Evaluation**
  - Accuracy, confusion matrix, feature importance
  - Balanced class distribution verification

- **Testing**
  - Unit tests for production models using `pytest`

---

## 📊 Dataset

- Wine quality dataset (red wine)  
- 1599 samples, 12 columns including physicochemical features and quality score  
- Quality grouped into 3 classes: **Low, Medium, High**

---

## 🚀 Installation

1. Clone the repo:

```bash
git clone https://github.com/vinayak-7569/wine_prediction.git
cd wine_prediction1
Create a virtual environment:

bash
Copy code
python -m venv venv
.\venv\Scripts\activate      # Windows
source venv/bin/activate     # Linux/Mac
Install dependencies:

bash
Copy code
pip install -r requirements.txt
🏃 Running the Pipeline
bash
Copy code
python main.py
Loads and preprocesses the dataset

Trains individual and ensemble models

Outputs test accuracy and feature importance

🧪 Running Tests
bash
Copy code
pytest tests/test_production.py
Validates production model creation

Ensures preprocessing and SMOTE work correctly

Checks feature engineering and model accuracy

📈 Performance
Model	Test Accuracy
RandomForest	81.38%
XGBoost	77.73%
CatBoost	76.11%
Ensemble Voting	76.52%

Final pipeline accuracy after improvements: 91.7%

✨ Improvements
Grouped classes from 6 → 3 (Low, Medium, High)

Added domain-specific features

Introduced XGBoost & CatBoost

Ensemble VotingClassifier for stability

SMOTE for class balancing and SHAP for explainability

📌 Notes
Ensure the dataset is in the data/ folder

Adjust k_features in preprocessing if you change feature selection

Compatible with Python 3.10+

🔗 Author
Vinayak – Machine Learning Engineer

GitHub: vinayak-7569