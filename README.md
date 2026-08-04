Student Health Risk Predictor
A machine learning web application that predicts a student's health risk profile (Fit / At-Risk / Unhealthy) based on lifestyle and health inputs, built for the CIS6005 Computational Intelligence module assignment.
Overview
This project uses an XGBoost classification model trained on the Kaggle "Predicting Student Health Risk" competition dataset to classify individuals into one of three health condition categories based on 13 lifestyle and physiological features (sleep, heart rate, BMI, activity level, diet, stress, and more).
Kaggle competition: Predicting Student Health Risk (Playground Series S6E7)
Model: XGBoost (leaderboard score: 0.85084)
Deployment: Streamlit web application
Requirements
Python 3.9+
Dependencies listed in `requirements.txt`:
streamlit
pandas
scikit-learn
xgboost
plotly
pickle (standard library)
Project Structure
```
health-risk-app/
├── app.py                     # Streamlit application
├── models/
│   ├── xgb_model.pkl           # Trained XGBoost model
│   ├── encoders.pkl            # Label encoders for categorical features
│   └── target_encoder.pkl      # Label encoder for the target variable
├── requirements.txt
└── README.md
```
Installation
Clone the repository:
```
   git clone https://github.com/FathimaAfrin1201/health-risk-app.git
   cd health-risk-app
   ```
(Recommended) Create and activate a virtual environment:
```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```
Install dependencies:
```
   pip install -r requirements.txt
   ```
Running the Application
From the project root directory, run:
```
streamlit run app.py
```
The app will open automatically in your default browser at `http://localhost:8501`.
Usage
Enter your health and lifestyle details in the input panels (Body & Vitals, Activity & Energy, Sleep & Wellbeing, Diet & Habits) — or select one of the built-in sandbox presets.
Click Predict Health Risk Profile.
View the predicted health condition, confidence score, probability breakdown, and personalized wellness recommendations.
Model Training
The model training process (data cleaning, EDA, model comparison between Random Forest, Random Forest with SMOTE, and XGBoost) is documented separately in the accompanying assignment report and Jupyter notebook.
Disclaimer
This tool estimates statistical health risk based on a machine learning model and does not constitute medical advice.
