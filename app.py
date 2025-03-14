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
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_absolute_error, r2_score
from imblearn.over_sampling import SMOTE

# Streamlit App Title with Themed Styling
st.set_page_config(page_title="Water Quality Prediction", page_icon="💧", layout="wide")
st.markdown("""
    <style>
    body {background-color: #f4f4f4; color: black;}
    .stTitle {color: black;}
    .stSidebar {background-color: #2E4053; color: white;}
    .stDataframe {background-color: white; color: black;}
    </style>
""", unsafe_allow_html=True)

st.title("💧 Water Quality Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

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
        df_cleaned.fillna(df_cleaned.median(numeric_only=True), inplace=True)

        label_encoder = LabelEncoder()
        df_cleaned[target] = label_encoder.fit_transform(df_cleaned[target])

        return df_cleaned, features, target, label_encoder
    else:
        return None, None, None, None

# Data Review Section
def data_review(df_cleaned):
    """Displays dataset overview."""
    st.subheader("📊 Data Review")
    st.write("First 5 rows of the dataset:")
    st.dataframe(df_cleaned.head())
    
    st.write("Dataset Information:")
    st.text(df_cleaned.info())

# Load the data
df_cleaned, features, target, label_encoder = load_data(uploaded_file)
if df_cleaned is not None:
    data_review(df_cleaned)
    
    X = df_cleaned[features]
    y = df_cleaned[target]

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
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Gaussian Naive Bayes': GaussianNB(),
            'Bagging': BaggingClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        accuracies = {}
        cv_accuracies = {}
        precision_scores = {}
        f1_scores = {}
        r2_scores = {}
        mae_scores = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracies[name] = accuracy_score(y_test, y_pred)
            cv_accuracies[name] = np.mean(cross_val_score(model, X_train, y_train, cv=5))
            precision_scores[name] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            f1_scores[name] = f1_score(y_test, y_pred, average='weighted')
            r2_scores[name] = r2_score(y_test, y_pred)
            mae_scores[name] = mean_absolute_error(y_test, y_pred)

        best_model_name = max(accuracies, key=accuracies.get)
        return models[best_model_name], best_model_name, accuracies, cv_accuracies, precision_scores, f1_scores, r2_scores, mae_scores

    best_model, best_model_name, accuracies, cv_accuracies, precision_scores, f1_scores, r2_scores, mae_scores = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    # Graphs and Metrics Display
    st.subheader("📈 Model Performance Comparison")
    metrics_df = pd.DataFrame({
        'Accuracy': accuracies,
        'Cross-Validation Accuracy': cv_accuracies,
        'Precision': precision_scores,
        'F1 Score': f1_scores,
        'R2 Score': r2_scores,
        'Mean Absolute Error': mae_scores
    })

    st.bar_chart(metrics_df)
    st.dataframe(metrics_df)

    st.subheader("🔮 Predict Water Quality")
    user_inputs = [st.number_input(f"{feature}", value=0.0) for feature in features]
    if st.button("Predict"):
        input_scaled = scaler.transform([user_inputs])
        prediction = best_model.predict(input_scaled)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        st.success(f"🔍 Predicted Water Quality: {predicted_label}")
