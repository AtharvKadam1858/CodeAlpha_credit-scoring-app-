import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Hide warnings

# Title
st.title("🧠 Credit Scoring Prediction App")
st.write("This Streamlit app predicts the credit risk using a machine learning model trained on financial data.")

# File upload
uploaded_file = st.file_uploader("📤 Upload your CSV file (e.g., cs-training.csv)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.write(data.head())

    # Drop missing values
    data.dropna(inplace=True)

    # Encode categorical columns
    le = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        data[col] = le.fit_transform(data[col])

    # Feature Engineering
    if 'RevolvingUtilizationOfUnsecuredLines' in data.columns and 'DebtRatio' in data.columns:
        data['Utilization_to_DebtRatio'] = data['RevolvingUtilizationOfUnsecuredLines'] / (data['DebtRatio'] + 1)

    # Target Identification
    target_candidates = [col for col in data.columns if col.lower() in ['class', 'target', 'seriousdlqin2yrs']]
    target_column = target_candidates[0] if target_candidates else data.columns[0]
    st.success(f"🎯 Detected target column: `{target_column}`")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model selection
    model_name = st.selectbox("🧠 Choose a Model", ['Logistic Regression', 'Decision Tree', 'Random Forest'])
    if model_name == 'Logistic Regression':
        model = LogisticRegression()
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)

    # Show results
    st.subheader("📈 Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("📉 ROC-AUC Score")
    st.success(f"AUC: {roc_auc:.2f}")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    st.pyplot()

    st.info("✅ Upload your own data and retrain the model live!")

else:
    st.warning("📂 Please upload a valid credit dataset CSV file.")