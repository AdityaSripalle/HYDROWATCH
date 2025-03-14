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
from sklearn.linear_model import LogisticRegression  # Corrected import for Logistic Regression
from sklearn.linear_model import SGDClassifier
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

# Train multiple models with cross-validation accuracy
def train_models(X_train, y_train, X_test, y_test):
    """Trains multiple models and selects the best one based on cross-validation accuracy."""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=20,
                                                min_samples_leaf=10, max_features='sqrt', random_state=42),
        "SVM": SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(criterion='gini', random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression - Sklearn": LogisticRegression(max_iter=1000),  # Corrected Logistic Regression
        "SGD Classifier": SGDClassifier(loss="log_loss", max_iter=1000, learning_rate='optimal', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    results = []

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        # Cross-validation accuracy on train data
        cv_accuracy = np.mean(cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy'))
        
        model.fit(X_train, y_train)  # Train final model
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)  # Actual test accuracy

        results.append([name, round(cv_accuracy, 4), round(test_acc, 4)])

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results, columns=["Model", "Cross-Validation Accuracy", "Testing Accuracy"])
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

        # Handle class imbalance using SMOTE **before splitting** (Prevents data leakage)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # ** 70-30 Split (Balanced Data) **
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

        # Train models and get the best one
        best_model, best_model_name, results_df = train_models(X_train, y_train, X_test, y_test)

        # Display Model Performance
        st.write("### 🔥 Model Performance")
        st.dataframe(results_df)

        st.write(f"✅ **Best Model Selected:** {best_model_name}")

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

            water_quality_mapping = {
                "Drinking Water": "✅ Safe for Drinking",
                "Irrigation": "🌾 Suitable for Irrigation",
                "Both": "💧 Safe for Both Drinking & Irrigation",
                "Harmful": "⚠️ Not Safe for Drinking"
            }

            result_text = water_quality_mapping.get(predicted_label, "⚠️ Unknown Classification")

            st.success(f"🔍 **Prediction:** {predicted_label} - {result_text}")
