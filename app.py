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

# Load dataset function
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    features = ['pH', 'EC', 'CO3', 'HCO3', 'Cl', 'SO4', 'NO3', 'TH', 'Ca', 'Mg', 'Na', 'K', 'F', 'TDS']
    target = 'Water Quality Classification'
    
    df_cleaned = df[features + [target]].copy()
    
    # Fill missing values efficiently
    df_cleaned.fillna(df_cleaned.median(), inplace=True)

    # Encode target variable
    label_encoder = LabelEncoder()
    df_cleaned[target] = label_encoder.fit_transform(df_cleaned[target])

    return df_cleaned, features, target, label_encoder

# Train multiple models and measure performance
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "SVM": SVC(kernel='rbf', random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
        "Bagging": BaggingClassifier(random_state=42, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(random_state=42),
    }
    
    results = []
    best_model = None
    best_accuracy = 0
    best_model_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        cv_scores = cross_val_score(model, X_train, y_train, cv=3, n_jobs=-1)
        mean_cv_accuracy = np.mean(cv_scores)

        results.append([name, train_acc, test_acc, mean_cv_accuracy])

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results, columns=["Model", "Training Accuracy", "Testing Accuracy", "CV Accuracy"])
    return best_model, best_model_name, results_df

# Load dataset
file_path = "DataSet.csv"
df, features, target, label_encoder = load_data(file_path)

# Preprocessing
X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance only if necessary
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

# Train models and get results
best_model, best_model_name, results_df = train_models(X_train, X_test, y_train, y_test)

# Save best model
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# Streamlit UI
st.title("💧 Water Quality Prediction App 🚀")
st.markdown("### Enter Water Parameters to Predict Water Quality")

# User Input
user_input = []
for feature in features:
    user_input.append(st.number_input(f"Enter {feature}", value=0.0))

if st.button("Predict Water Quality"):
    user_input_scaled = scaler.transform([user_input])
    prediction = best_model.predict(user_input_scaled)
    class_label = label_encoder.inverse_transform(prediction)[0]
    st.success(f"🏆 Predicted Water Quality: **{class_label}**")

st.markdown("### 🔥 Model Performance")
st.dataframe(results_df)
st.write(f"✅ Best Model: **{best_model_name}** with **{results_df['Testing Accuracy'].max():.2%}** accuracy")
