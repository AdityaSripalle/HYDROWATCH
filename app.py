import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

st.title("💧 Water Quality Prediction (Optimized)")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

@st.cache_data
def load_data(uploaded_file):
    """Loads and processes the dataset."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        features = ['pH', 'EC', 'CO3', 'HCO3', 'Cl', 'SO4', 'NO3', 'TH', 'Ca', 'Mg', 'Na', 'K', 'F', 'TDS', 'WQI']
        target = 'Water Quality Classification'

        if target not in df.columns:
            st.error("❌ Error: The dataset does not contain the expected target column.")
            return None, None, None, None

        df_cleaned = df[features + [target]].copy()
        df_cleaned.fillna(df_cleaned.median(numeric_only=True), inplace=True)  # Handle missing values

        label_encoder = LabelEncoder()
        df_cleaned[target] = label_encoder.fit_transform(df_cleaned[target])

        return df_cleaned, features, target, label_encoder
    return None, None, None, None

# Train multiple models with overfitting prevention
def train_models(X_train, y_train):
    """Trains multiple models and selects the best one based on accuracy using cross-validation.""
