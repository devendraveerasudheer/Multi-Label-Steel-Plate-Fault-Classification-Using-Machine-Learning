import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Steel Plate Fault Detection",
    page_icon="🔍",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
with open("model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
feature_names = model_data["feature_names"]
fault_names = model_data["fault_names"]


# -----------------------------
# Title
# -----------------------------
st.title("🔍 Steel Plate Fault Detection")

st.write(
    "Enter the values for the 27 features below to predict "
    "the possible steel plate faults."
)


# -----------------------------
# Input Features
# -----------------------------
st.subheader("Enter Feature Values")

input_data = {}

# Create 3 columns for better layout
cols = st.columns(3)

for i, feature in enumerate(feature_names):

    with cols[i % 3]:
        input_data[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.4f"
        )


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Fault", use_container_width=True):

    # Convert input into DataFrame
    input_df = pd.DataFrame(
        [input_data],
        columns=feature_names
    )

    # Prediction
    prediction = model.predict(input_df)

    # Convert from [[0, 1, ...]]
    prediction = prediction[0]

    st.subheader("Prediction Result")

    detected_faults = []

    for i, fault in enumerate(fault_names):

        if prediction[i] == 1:
            detected_faults.append(fault)

    # -----------------------------
    # Display Results
    # -----------------------------

    if detected_faults:

        st.success("Fault(s) Detected")

        for fault in detected_faults:
            st.write("✅", fault)

    else:

        st.info("No fault detected.")