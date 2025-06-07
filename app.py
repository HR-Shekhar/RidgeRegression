# app.py
import streamlit as st
import numpy as np
import joblib

# Load both models
scratch_model = joblib.load("ridge_scratch.pkl")  # {'w': ..., 'b': ...}
sk_model = joblib.load("ridge_sklearn.pkl")       # Ridge()

# Page config
st.set_page_config(page_title="Ridge Regression Comparison", page_icon="🔧", layout="centered")
st.title("🔧 Ridge Regression: Scratch vs scikit-learn")

st.markdown("""
This app compares **Ridge Regression predictions** for multiple features, using:
- 🤖 A model built from scratch
- 🧠 A model trained with `scikit-learn`

Enter the feature values below:
""")

# Sample input features — customize according to your dataset
feature_names = ["Engine Size (L)", "Cylinders", "Fuel Consumption City (L/100km)"]

# Take inputs
input_values = []
for feature in feature_names:
    val = st.number_input(feature, value=0.0, step=0.1)
    input_values.append(val)

x_input = np.array(input_values)           # shape: (n_features,)
x_input_2D = x_input.reshape(1, -1)         # for sklearn model

if st.button("Predict with Both Models"):
    # Scratch model
    w, b = scratch_model["w"], scratch_model["b"]
    pred_scratch = np.dot(w, x_input) + b

    # scikit-learn model
    pred_sklearn = sk_model.predict(x_input_2D)[0]

    # Output results
    st.subheader("📊 Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Scratch Model", f"{pred_scratch:.2f}")
    with col2:
        st.metric("scikit-learn Model", f"{pred_sklearn:.2f}")

# Footer
st.markdown("---")
st.markdown("Built by **Himanshu Shekhar** — Ridge Regression with multiple features 🚀")
