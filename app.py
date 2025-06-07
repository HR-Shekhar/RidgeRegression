
import streamlit as st
import numpy as np
import joblib

# Load both models
scratch_model = joblib.load("ridge_scratch.pkl")  # {'w': ..., 'b': ..., 'scaler': ...}
sklearn_model = joblib.load("ridge_sklearn.pkl")  # Ridge model already trained

# Extract scratch model parameters
w = scratch_model["w"]
b = scratch_model["b"]
scaler = scratch_model["scaler"]

# App UI
st.set_page_config(page_title="Compare Ridge Models", page_icon="📊", layout="centered")
st.title("📊 Insurance Charges Prediction: Scratch vs Scikit-learn")

st.markdown("""
Compare predicted insurance charges based on:
- **Age**
- **BMI**
- **Number of Children**
- **Smoker Status**

Models used:
- 💻 Ridge Regression (from scratch)
- 🤖 Ridge Regression (scikit-learn)
""")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=1, step=1)
smoker = st.selectbox("Smoker", ["No", "Yes"])
smoker_encoded = 1 if smoker == "Yes" else 0

# Predict on button click
if st.button("Predict Charges"):
    x_input = np.array([[age, bmi, children, smoker_encoded]])

    # Scale using scratch model's scaler
    x_scaled = scaler.transform(x_input)

    # Scratch prediction
    pred_scratch = np.dot(x_scaled, w) + b

    # Scikit-learn prediction
    pred_sklearn = sklearn_model.predict(x_scaled)[0]

    # Display
    st.subheader("🔍 Prediction Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Scratch Ridge", f"${pred_scratch[0]:,.2f}")
    with col2:
        st.metric("scikit-learn Ridge", f"${pred_sklearn:,.2f}")

st.markdown("---")
st.markdown("Built by **Himanshu Shekhar** | Ridge Regression Comparison 💡")
