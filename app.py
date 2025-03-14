import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif

# Load dataset function
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        features = ['pH', 'EC', 'CO3', 'HCO3', 'Cl', 'SO4', 'NO3', 'TH', 'Ca', 'Mg', 'Na', 'K', 'F', 'TDS']
        target = 'Water Quality Classification'

        if target not in df.columns:
            st.error("Error: The dataset does not contain the expected target column.")
            return None, None, None, None

        df_cleaned = df[features + [target]].copy()
        df_cleaned.fillna(df_cleaned.median(), inplace=True)

        label_encoder = LabelEncoder()
        df_cleaned[target] = label_encoder.fit_transform(df_cleaned[target])

        return df_cleaned, features, target, label_encoder
    else:
        return None, None, None, None

# Streamlit UI
st.title("💧 Water Quality Prediction App 🚀")
st.markdown("### Upload Your Dataset to Start")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df, features, target, label_encoder = load_data(uploaded_file)

    if df is not None:
        st.success("✅ Dataset Loaded Successfully!")
        st.write(df.head())  # Show first few rows

        # Preprocessing
        X = df[features]
        y = df[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Handle class imbalance if necessary
        if np.bincount(y).min() / np.bincount(y).max() < 0.5:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        else:
            X_resampled, y_resampled = X_scaled, y

        # Feature Selection
        selector = SelectKBest(score_func=f_classif, k=10)
        X_selected = selector.fit_transform(X_resampled, y_resampled)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_resampled, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        st.write(f"🔹 Training Accuracy: **{train_acc:.2%}**")
        st.write(f"🔹 Testing Accuracy: **{test_acc:.2%}**")

        # Save model
        joblib.dump(model, "best_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(label_encoder, "label_encoder.pkl")

        # Prediction UI
        st.markdown("### Enter Water Parameters to Predict Quality")
        user_input = []
        for feature in features:
            user_input.append(st.number_input(f"Enter {feature}", value=0.0))

        if st.button("Predict Water Quality"):
            user_input_scaled = scaler.transform([user_input])
            prediction = model.predict(user_input_scaled)
            class_label = label_encoder.inverse_transform(prediction)[0]
            st.success(f"🏆 Predicted Water Quality: **{class_label}**")
