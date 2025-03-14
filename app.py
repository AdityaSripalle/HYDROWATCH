import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE

# Title for the Streamlit app
st.title("💧 Water Quality Prediction (Optimized)")

# Upload CSV file
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

        X = df_cleaned[features]
        y = df_cleaned[target]

        return X, y, df_cleaned, label_encoder
    else:
        return None, None, None, None

# Load the data
X, y, df_cleaned, label_encoder = load_data(uploaded_file)

if X is not None and y is not None:
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    def train_and_evaluate_models(X_train, y_train, X_test, y_test):
        """Trains multiple models and selects the best one based on testing accuracy."""
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', C=1, random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski'),
            'Decision Tree': DecisionTreeClassifier(criterion='gini', random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Gaussian Naive Bayes': GaussianNB(),
            'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42),
            'AdaBoost': AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        accuracies = {}
        cv_accuracies = {}
        precision_scores = {}
        f1_scores = {}

        # Initialize Stratified K-Fold
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            # Perform cross-validation
            cv_score = cross_val_score(model, X_train, y_train, cv=stratified_kfold, scoring='accuracy')
            cv_accuracies[name] = cv_score.mean()

            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Compute performance metrics
            accuracies[name] = accuracy_score(y_test, y_pred)
            precision_scores[name] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            f1_scores[name] = f1_score(y_test, y_pred, average='weighted')

        # Find the best model based on testing accuracy
        best_model_name = max(accuracies, key=accuracies.get)
        return models[best_model_name], best_model_name, accuracies, cv_accuracies, precision_scores, f1_scores

    # Train and evaluate models
    best_model, best_model_name, accuracies, cv_accuracies, precision_scores, f1_scores = train_and_evaluate_models(
        X_train, y_train, X_test, y_test
    )

    # Display the comparison bar chart of model accuracies
    st.title("Comparison of Model Accuracies")

    st.subheader("Comparison of Cross-Validation Accuracies")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(cv_accuracies.keys(), cv_accuracies.values(), color='skyblue')
    ax.set_xlabel('Models', fontsize=14)
    ax.set_ylabel('Cross-Validation Accuracy', fontsize=14)
    ax.set_title('Comparison of Cross-Validation Accuracies', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    st.pyplot(fig)

    st.subheader("Comparison of Testing Accuracies")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(accuracies.keys(), accuracies.values(), color='lightgreen')
    ax.set_xlabel('Models', fontsize=14)
    ax.set_ylabel('Testing Accuracy', fontsize=14)
    ax.set_title('Comparison of Testing Accuracies', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    st.pyplot(fig)

    # Display Precision and F1 Score metrics comparison
    st.title("Model Performance Metrics Comparison")
    st.subheader("Comparison of Precision and F1 Score")

    metrics_df = pd.DataFrame({
        'Precision': precision_scores,
        'F1 Score': f1_scores
    })

    st.dataframe(metrics_df)

    fig, ax = plt.subplots(figsize=(14, 8))
    metrics_df.plot(kind='bar', ax=ax)
    ax.set_title("Comparison of Model Metrics", fontsize=16)
    ax.set_xlabel("Models", fontsize=14)
    ax.set_ylabel("Metric Values", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    st.pyplot(fig)

    # Display the best model
    st.success(f"🎉 Best Model: {best_model_name} with Accuracy: {accuracies[best_model_name] * 100:.2f}%")


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
