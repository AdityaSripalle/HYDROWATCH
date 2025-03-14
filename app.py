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

# Streamlit UI setup
st.set_page_config(page_title="Water Quality Prediction", layout="wide")
st.title("💧 Water Quality Prediction App")
st.markdown("This app allows users to train machine learning models to classify water quality based on different parameters.")

# Load dataset
def load_data(file):
    df = pd.read_csv(file)
    features = ['pH', 'EC', 'CO3', 'HCO3', 'Cl', 'SO4', 'NO3', 'TH', 'Ca', 'Mg', 'Na', 'K', 'F', 'TDS']
    target = 'Water Quality Classification'
    df_cleaned = df[features + [target]].copy()
    numeric_features = df_cleaned.select_dtypes(include=np.number).columns
    df_cleaned[numeric_features] = df_cleaned[numeric_features].fillna(df_cleaned[numeric_features].median())
    categorical_features = df_cleaned.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        df_cleaned[feature].fillna(df_cleaned[feature].mode()[0], inplace=True)
    label_encoder = LabelEncoder()
    df_cleaned[target] = label_encoder.fit_transform(df_cleaned[target])
    return df_cleaned, features, target, label_encoder

# Train models
def train_models(X, y):
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
        scores = cross_val_score(model, X, y, cv=5)
        mean_accuracy = np.mean(scores)
        results.append([name, mean_accuracy])
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_model = model
            best_model_name = name
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
    return best_model, best_model_name, results_df

# Main Streamlit app logic
uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    df, features, target, label_encoder = load_data(uploaded_file)
    st.write("### Preview of Dataset")
    st.dataframe(df.head())
    
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
    
    st.write("### Training Models...")
    best_model, best_model_name, results_df = train_models(X_train, y_train)
    st.dataframe(results_df)
    st.success(f"Best Model: {best_model_name}")
    
    # Train best model
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    st.write("### Model Performance")
    st.text(classification_report(y_test, y_pred))
    
    # Save models
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    
    # User input for prediction
    st.write("## Predict Water Quality")
    user_input = []
    for feature in features:
        user_input.append(st.number_input(f"Enter {feature}", min_value=0.0, step=0.01))
    
    if st.button("Predict"):
        user_input_scaled = scaler.transform([user_input])
        prediction = best_model.predict(user_input_scaled)
        class_label = label_encoder.inverse_transform(prediction)[0]
        st.success(f"Predicted Water Quality: {class_label}")
