import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("linear_regression_model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------------
# Streamlit App Title and Description
# -------------------------------
st.title("🏥 Medical Insurance Charges Predictor")

st.write("""
This app predicts **Medical Insurance Charges** based on:
- **Age**
- **BMI**
- **Smoking Status**

Built using **Linear Regression** and **Streamlit**.
""")

# -------------------------------
# Collect User Inputs
# -------------------------------
age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("Enter BMI", min_value=10.0, max_value=60.0, value=25.0)
smoker = st.selectbox("Are you a smoker?", ["No", "Yes"])

# Encode smoker variable (same as in training)
smoker_yes = 1 if smoker == "Yes" else 0

# Prepare input for prediction
input_data = np.array([[age, bmi, smoker_yes]])

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Charges"):
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Medical Charges: **${prediction[0]:,.2f}**")

# -------------------------------
# Footer Section with Links
# -------------------------------
st.markdown("---")
st.markdown("### 📂 Project Info")
st.markdown("""
**GitHub Repository:**(https://github.com/krushna8767/Insurance-Charges-Predictor)

📧 **Contact:** (mailto: kshinde.876732@gmail.com)

Made with ❤️ using [Streamlit](https://streamlit.io)
""")



