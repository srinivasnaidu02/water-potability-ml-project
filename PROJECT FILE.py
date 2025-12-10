import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Water Potability Predictor", layout="centered")
st.title("💧 Water Potability Prediction")
st.write("Prediction based on *water quality metrics*.")

# ------------------------------------------------------
# 1. LOAD AND PREPROCESS DATA
# ------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    # Handle missing values by filling with mean
    df["ph"] = df["ph"].fillna(df["ph"].mean())
    df["Sulfate"] = df["Sulfate"].fillna(df["Sulfate"].mean())
    df["Trihalomethanes"] = df["Trihalomethanes"].fillna(df["Trihalomethanes"].mean())
    return df

df = load_data()

# ------------------------------------------------------
# 2. TRAIN MODEL
# ------------------------------------------------------
@st.cache_resource
def train_model(df):
    X = df.drop(columns=["Potability"])
    y = df["Potability"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns

model, feature_names = train_model(df)

# ------------------------------------------------------
# 3. USER INPUT UI
# ------------------------------------------------------
st.subheader("Enter Water Quality Parameters")

col1, col2 = st.columns(2)

with col1:
    ph = st.number_input("pH level (0-14)", 0.0, 14.0, 7.0)
    Hardness = st.number_input("Hardness (mg/L)", 0.0, 400.0, 150.0)
    Solids = st.number_input("Solids (ppm)", 0.0, 60000.0, 20000.0)
    Chloramines = st.number_input("Chloramines (ppm)", 0.0, 15.0, 7.0)
    Sulfate = st.number_input("Sulfate (mg/L)", 0.0, 500.0, 300.0)

with col2:
    Conductivity = st.number_input("Conductivity (μS/cm)", 0.0, 800.0, 400.0)
    Organic_carbon = st.number_input("Organic_carbon (ppm)", 0.0, 30.0, 15.0)
    Trihalomethanes = st.number_input("Trihalomethanes (μg/L)", 0.0, 150.0, 60.0)
    Turbidity = st.number_input("Turbidity (NTU)", 0.0, 10.0, 4.0)

user_input = pd.DataFrame([{
    "ph": ph,
    "Hardness": Hardness,
    "Solids": Solids,
    "Chloramines": Chloramines,
    "Sulfate": Sulfate,
    "Conductivity": Conductivity,
    "Organic_carbon": Organic_carbon,
    "Trihalomethanes": Trihalomethanes,
    "Turbidity": Turbidity
}])

# ------------------------------------------------------
# 4. PREDICT BUTTON
# ------------------------------------------------------
if st.button("Predict Potability"):
    prediction = model.predict(user_input)[0]
    prob = model.predict_proba(user_input)[0][1] * 100

    if prediction == 1:
        st.success(f"✅ The water is Potable (Safe to drink)\nProbability: {prob:.2f}%")
    else:
        st.error(f"⚠ The water is NOT Potable (Unsafe)\nProbability: {100-prob:.2f}% (Unsafe)")

# ------------------------------------------------------
# 5. OPTIONAL: SHOW DATASET STATS
# ------------------------------------------------------
with st.expander("📊 Show Dataset Statistics"):
    st.write(df.describe())
    
    st.write("Potability Distribution:")
    fig, ax = plt.subplots()
    df["Potability"].value_counts().plot(kind="bar", ax=ax, color=["red", "green"])
    ax.set_xticklabels(["Not Potable", "Potable"], rotation=0)
    st.pyplot(fig)


