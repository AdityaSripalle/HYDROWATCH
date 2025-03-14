import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif

# Streamlit app title
st.title("💧 Water Quality Classification")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

@st.cache_data
def load_data(uploaded_file):
    """Loads and processes the dataset."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        features = ['pH', 'EC', 'CO3', 'HCO3', 'Cl', 'SO4', 'NO3', 'TH', 'Ca', 'Mg', 'Na', 'K', 'F', 'TDS']
        target = 'Water Quality Classification'

        if target not in df.columns:
            st.error("❌ Error: The dataset does not contain the expected target column.")
            return None, None, None, None

        df_cleaned = df[features + [target]].copy()

        # Handle missing values
        numeric_features = df_cleaned.select_dtypes(include=[np.number]).columns
        categorical_features = df_cleaned.select_dtypes(exclude=[np.number]).columns

        df_cleaned[numeric_features] = df_cleaned[numeric_features].fillna(df_cleaned[numeric_features].median())
        for feature in categorical_features:
            df_cleaned[feature].fillna(df_cleaned[feature].mode()[0], inplace=True)

        # Encode target labels
        label_encoder = LabelEncoder()
        df_cleaned[target] = label_encoder.fit_transform(df_cleaned[target])

        return df_cleaned, features, target, label_encoder
    return None, None, None, None

# Train multiple models
def train_models(X_train, y_train, X_test, y_test):
    """Trains multiple ML models and returns the best model with results."""
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(kernel='rbf', random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
        "Bagging": BaggingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42)
    }

    results = []
    best_model = None
    best_accuracy = 0
    best_model_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        cv_acc = np.mean(cross_val_score(model, X_train, y_train, cv=5))

        results.append([name, cv_acc, train_acc, test_acc])

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results, columns=["Model", "CV Accuracy", "Training Accuracy", "Testing Accuracy"])
    return best_model, best_model_name, results_df

# Predict water quality
def predict_water_quality(model, scaler, label_encoder, input_data):
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    return label_encoder.inverse_transform(prediction)[0]

if uploaded_file:
    df, features, target, label_encoder = load_data(uploaded_file)
    
    if df is not None:
        st.write("### 📊 Preview of the Dataset")
        st.dataframe(df.head())

        # Data Preprocessing
        X = df[features]
        y = df[target]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=10)
        X_selected = selector.fit_transform(X_resampled, y_resampled)

        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_resampled, test_size=0.2, random_state=42)

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

        # Prediction Form
        st.write("### 🔮 Predict Water Quality")
        user_inputs = []
        for feature in features:
            user_inputs.append(st.number_input(f"Enter {feature}", value=0.0))

        if st.button("Predict"):
            prediction = predict_water_quality(best_model, scaler, label_encoder, user_inputs)
            st.success(f"Predicted Water Quality: {prediction}")
