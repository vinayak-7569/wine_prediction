# 🍷 Wine Quality Prediction (Classification Model)

This project predicts wine quality using advanced feature engineering and classification models. It groups wine quality into three classes — Low, Medium, and High — and uses Random Forest as the final production model.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/vinayak-7569/wine_prediction.git
cd wine_prediction
2. Create and Activate a Virtual Environment
bash
Copy
Edit
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Run the Main Pipeline
bash
Copy
Edit
python main.py
✅ Expected Output
yaml
Copy
Edit
Starting Wine Quality Prediction Pipeline...
Accuracy: 0.8688 (86.88%)
Model saved: models/production_wine_model.joblib
Predictor code saved: models/wine_predictor.py
📊 Model Performance
Model	Accuracy	Notes
Random Forest	86.88%	Final selected model
Baseline Model	~70.00%	Before feature engineering

🧠 Key Features
🔧 Feature Engineering
alcohol_sulphates: Interaction of alcohol and sulphates

acid_ratio: Balance of fixed and volatile acidity

alcohol_quality_proxy: Alcohol scaled with quality indicator

🎯 Classification Approach
Low Quality: 3–4

Medium Quality: 5–6

High Quality: 7–8

→ Imbalance addressed using class_weight='balanced'

📦 Production Artifacts
models/production_wine_model.joblib → Final trained model

models/wine_predictor.py → Inference-ready predictor

models/feature_names.joblib → Features used for training

⚙️ Technical Stack
Python 3.10 / 3.13+

pandas 2.2.3

numpy 1.26.4

scikit-learn 1.5.2

joblib 1.4.2

seaborn, matplotlib (for EDA/visualization)

🌱 Future Enhancements
✅ SHAP-based model explainability

✅ Hyperparameter tuning (GridSearchCV)

✅ Add XGBoost and CatBoost comparisons

✅ Deploy via Streamlit or Flask web app

📝 License
This project is licensed under the MIT License.

👤 Contact
GitHub: @vinayak-7569

Project Repo: Wine Quality Prediction

yaml
Copy
Edit

---

### 📌 Instructions:

1. Copy the content above into a `README.md` file inside your repo.
2. Commit and push:
   ```bash
   git add README.md
   git commit -m "Add polished README"
   git push origin main
