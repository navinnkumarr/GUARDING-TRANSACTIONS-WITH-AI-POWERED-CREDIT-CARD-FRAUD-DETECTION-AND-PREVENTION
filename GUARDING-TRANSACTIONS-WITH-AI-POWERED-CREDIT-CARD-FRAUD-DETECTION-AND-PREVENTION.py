import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import requests
import io

# Set random seed for reproducibility
np.random.seed(42)

# Streamlit app title
st.title("Fraud Detection with Random Forest")

# Display the updated date and time
st.write("**Date and Time:** 02:44 PM IST on Wednesday, May 14, 2025")

# Sidebar for user interaction
st.sidebar.header("Model Controls")
train_button = st.sidebar.button("Train Model")

# Step 1: Load dataset from a cloud storage URL with API key
st.header("Dataset from Cloud Storage (API Key Protected)")
# Replace this base URL with the URL of your dataset (without the API key)
base_url = "YOUR_CLOUD_STORAGE_BASE_URL_HERE"  # Example: https://drive.google.com/uc?export=download&id=FILE_ID

# Access the API key from Streamlit secrets
try:
    api_key = st.secrets["API_KEY"]
except KeyError:
    st.error("API_KEY not found in Streamlit secrets. Please set it in .streamlit/secrets.toml or Streamlit Cloud settings.")
    st.stop()

# Construct the full URL with the API key
data_url = f"{base_url}?api_key={api_key}"

# Fetch the dataset using requests
try:
    response = requests.get(data_url)
    response.raise_for_status()  # Raise an error for bad status codes
    # Load the dataset into a Pandas DataFrame from the response content
    df = pd.read_csv(io.BytesIO(response.content))
    st.dataframe(df)
except requests.exceptions.RequestException as e:
    st.error(f"Error fetching dataset: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading dataset into DataFrame: {e}")
    st.stop()

# Step 2: Data Preprocessing and Model Training
if train_button:
    st.header("Model Training and Evaluation")
    
    # Preprocessing
    X = df.drop('Class', axis=1)
    y = df['Class']
    scaler = StandardScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    # Model Evaluation
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Accuracy")
    st.write(f"{accuracy:.4f}")
    
    roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    st.subheader("ROC AUC Score")
    st.write(f"{roc_auc:.4f}")
    
    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    st.pyplot(fig)
    
    # Precision-Recall Curve
    st.subheader("Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label='Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    st.pyplot(fig)
    
    # Save model and scaler
    joblib.dump(rf_model, 'fraud_detection_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    st.success("Model and scaler saved successfully!")
