"""
GUARDING TRANSACTIONS WITH AI-POWERED CREDIT CARD FRAUD DETECTION AND PREVENTION

This script demonstrates how to implement credit card fraud detection using machine learning in Python.
It covers data loading, exploration, preprocessing, model training, evaluation, and visualization.

Dataset used: The Credit Card Fraud Detection dataset from Kaggle:
https://www.kaggle.com/mlg-ulb/creditcardfraud
(Please download and place the 'creditcard.csv' file in the same directory as this script.)

Requirements:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn (for handling imbalanced dataset)
- optionally xgboost (for another model)

Install missing packages with:
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost

Usage:
python credit_card_fraud_detection.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

from imblearn.under_sampling import RandomUnderSampler

# Optional XGBoost import
try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False

def load_data(path="creditcard.csv"):
    print("Loading dataset...")
    data = pd.read_csv(path)
    print(f"Shape of data: {data.shape}")
    return data

def explore_data(data):
    print("\n----- Data Info -----")
    print(data.info())
    print("\n----- Statistical Summary -----")
    print(data.describe())

    print("\nDistribution of transaction classes:")
    print(data['Class'].value_counts())

    plt.figure(figsize=(6,4))
    sns.countplot(x='Class', data=data)
    plt.title("Class Distribution (0 = Legitimate, 1 = Fraud)")
    plt.show()

    print("\nTransaction amount statistics:")
    print(data.groupby('Class')['Amount'].describe())

    plt.figure(figsize=(8,4))
    sns.boxplot(x='Class', y='Amount', data=data)
    plt.title("Transaction Amount by Class")
    plt.show()

def preprocess_data(data):
    # The dataset's Time and Amount features require scaling
    print("Preprocessing data...")

    data = data.copy()
    scaler = StandardScaler()
    data['Amount_scaled'] = scaler.fit_transform(data['Amount'].values.reshape(-1,1))
    data['Time_scaled'] = scaler.fit_transform(data['Time'].values.reshape(-1,1))

    # Drop the original 'Amount' and 'Time' columns
    data.drop(['Amount', 'Time'], axis=1, inplace=True)

    X = data.drop('Class', axis=1)
    y = data['Class']

    return X, y

def handle_imbalance(X, y):
    print("Handling class imbalance with Random Under Sampling...")
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)

    print(f"Resampled dataset shape: {X_res.shape}, {y_res.shape}")
    print(f"Class distribution after resampling: {np.bincount(y_res)}")
    return X_res, y_res

def train_models(X_train, y_train):
    print("Training models...")
    models = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    # XGBoost if available
    if xgboost_available:
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb.fit(X_train, y_train)
        models['XGBoost'] = xgb

    return models

def evaluate_models(models, X_test, y_test):
    print("\nEvaluating models...\n")

    for name, model in models.items():
        print(f"--- {name} ---")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]

        print(classification_report(y_test, y_pred))
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC Score: {roc_auc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate','Fraud'])
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.show()

def main():
    data = load_data()
    explore_data(data)
    X, y = preprocess_data(data)
    X_res, y_res = handle_imbalance(X, y)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)

if __name__ == '__main__':
    main()

