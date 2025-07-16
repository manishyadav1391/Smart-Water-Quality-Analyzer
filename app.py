import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

st.set_page_config("Smart Water Quality Analyzer", layout="wide", page_icon="💧")
st.title("💧 Smart Water Quality Analyzer")

# Load dataset and model
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    df.fillna(df.mean(), inplace=True)
    return df

@st.cache_resource
def train_model():
    df = load_data()
    X = df.drop("Potability", axis=1)
    y = df["Potability"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_model()
df = load_data()


tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📊 Analyze Dataset", "🧪 Test Sample", "📈 Model Insights"])

# TAB 1: HOME
with tab1:
    st.markdown("### 🔬 Project Overview")
    st.write("This AI-powered tool predicts **whether a given water sample is safe to drink** using a machine learning model trained on real-world water quality data.")
    st.info("**Goal:** Support SDG 6 - Clean Water and Sanitation by providing an early water safety assessment tool.")
    st.markdown("#### 📁 Dataset Features")
    st.dataframe(df.head())

# TAB 2: DATA INSIGHTS
with tab2:
    st.markdown("### 📊 Dataset Analysis")
    st.write("Shape: ", df.shape)
    st.write("Missing Values:")
    st.write(df.isnull().sum())
    st.write("Feature Correlation:")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# TAB 3: TEST SAMPLE
with tab3:
    st.markdown("### 🧪 Water Sample Input")
    cols = df.columns[:-1]
    user_data = {}
    for col in cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        user_data[col] = st.slider(f"{col}", min_val, max_val, float(df[col].mean()))

    input_df = pd.DataFrame([user_data])
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][pred]
    st.success(f"**Prediction**: {'Safe ✅' if pred==1 else 'Unsafe ⚠️'} (Confidence: {prob*100:.2f}%)")

    if pred == 0:
        st.warning("💡 Tip: Boil or filter this water before drinking.")
    else:
        st.info("✅ Good to go! Still, regular testing is recommended.")

# TAB 4: MODEL INSIGHTS
with tab4:
    st.markdown("### 📈 Model Performance")
    X = df.drop("Potability", axis=1)
    y = df["Potability"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    preds = model.predict(X_test)
    st.code(classification_report(y_test, preds), language='text')
    st.write("ROC AUC Score:", roc_auc_score(y_test, preds))
