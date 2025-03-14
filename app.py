import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

st.set_page_config(page_title="Water Quality Prediction", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #30475e;
        color: white;
    }
    .stButton>button {
        background-color: #f05454;
        color: white;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("💧 Water Quality Prediction")
st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

def load_data(file):
    df = pd.read_csv(file)
    features = ['pH', 'EC', 'CO3', 'HCO3', 'Cl', 'SO4', 'NO3', 'TH', 'Ca', 'Mg', 'Na', 'K', 'F', 'TDS']
    target = 'Water Quality Classification'
    df_cleaned = df[features + [target]].copy()
    df_cleaned.fillna(df_cleaned.median(), inplace=True)
    label_encoder = LabelEncoder()
    df_cleaned[target] = label_encoder.fit_transform(df_cleaned[target])
    return df_cleaned, features, target, label_encoder

def train_models(X, y):
    models = {
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
        "Bagging": BaggingClassifier(),
        "AdaBoost": AdaBoostClassifier()
    }
    results = []
    best_model, best_accuracy, best_model_name = None, 0, ""
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5)
        mean_accuracy = np.mean(scores)
        results.append([name, mean_accuracy])
        if mean_accuracy > best_accuracy:
            best_accuracy, best_model, best_model_name = mean_accuracy, model, name
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)
    return best_model, best_model_name, results_df

def predict_water_quality(model, scaler, label_encoder, features):
    user_input = [st.number_input(f"Enter {feature}", value=0.0) for feature in features]
    if st.button("Predict Water Quality"):
        user_input_scaled = scaler.transform([user_input])
        prediction = model.predict(user_input_scaled)
        class_label = label_encoder.inverse_transform(prediction)[0]
        st.success(f"Predicted Water Quality: {class_label}")

if file:
    df, features, target, label_encoder = load_data(file)
    X, y = df[features], df[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_model, best_model_name, results_df = train_models(X_scaled, y)
    st.subheader("Model Performance")
    st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy'], color='lightgreen'))
    st.subheader(f"Best Model: {best_model_name}")
    best_model.fit(X_scaled, y)
    st.subheader("Make a Prediction")
    predict_water_quality(best_model, scaler, label_encoder, features)
