import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Streamlit App Title
st.title("💧 Water Quality Classification (Optimized)")

# File uploader
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

        # Handle missing values
        df_cleaned.fillna(df_cleaned.median(numeric_only=True), inplace=True)

        # Encode target labels
        label_encoder = LabelEncoder()
        df_cleaned[target] = label_encoder.fit_transform(df_cleaned[target])

        return df_cleaned, features, target, label_encoder
    return None, None, None, None

# Train models
def train_models(X_train, y_train, X_test, y_test):
    """Trains selected ML models and returns the best model."""
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(kernel='rbf', random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "KNN": KNeighborsClassifier()
    }

    results = []
    best_model = None
    best_accuracy = 0
    best_model_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        results.append([name, train_acc, test_acc])

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results, columns=["Model", "Training Accuracy", "Testing Accuracy"])
    return best_model, best_model_name, results_df

if uploaded_file:
    df, features, target, label_encoder = load_data(uploaded_file)
    
    if df is not None:
        st.write("### 📊 Dataset Preview")
        st.dataframe(df.head())

        # Data Preprocessing
        X = df[features]
        y = df[target]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Train models
        best_model, best_model_name, results_df = train_models(X_train, y_train, X_test, y_test)

        # Display Model Performance
        st.write("### 🔥 Model Performance")
        st.dataframe(results_df)

        st.write(f"### 🏆 Best Model: {best_model_name}")

        # Save the best model
        joblib.dump(best_model, "best_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(label_encoder, "label_encoder.pkl")

        # --- Prediction Form ---
        st.write("### 🔮 Predict Water Quality")

        with st.form(key="input_form"):
            user_inputs = [st.number_input(f"{feature}", value=0.0) for feature in features]
            submit_button = st.form_submit_button("Predict")

        if submit_button:
            input_scaled = scaler.transform([user_inputs])
            prediction = best_model.predict(input_scaled)
            predicted_label = label_encoder.inverse_transform(prediction)[0]

            # Map prediction to readable categories
            prediction_mapping = {
                "Drinking Water": "✅ Water is suitable for drinking",
                "Irrigation Water": "🚜 Water is suitable for irrigation",
                "Both Drinking & Irrigation": "🌿 Water is suitable for both drinking and irrigation",
                "Harmful Water": "❌ Water is harmful to drink"
            }

            result_text = prediction_mapping.get(predicted_label, "⚠️ Unknown Classification")
            st.success(f"🔍 **Prediction:** {result_text}")
