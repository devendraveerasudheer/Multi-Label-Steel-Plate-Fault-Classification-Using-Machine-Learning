import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Steel Plate Fault Detection",
    page_icon="🔍",
    layout="wide"
)

# Check sklearn
try:
    import sklearn
    st.success(f"scikit-learn version: {sklearn.__version__}")
except Exception as e:
    st.error("scikit-learn is not installed.")
    st.code(str(e))
    st.stop()

# Load model
try:
    with open("model.pkl", "rb") as file:
        model_data = pickle.load(file)
except Exception as e:
    st.error("Error loading model.pkl")
    st.code(str(e))
    st.stop()

model = model_data["model"]
feature_names = model_data["feature_names"]
fault_names = model_data["fault_names"]

st.title("🔍 Steel Plate Fault Detection")

st.write(
    "Enter the values for the steel plate features "
    "to predict the possible faults."
)

st.subheader("Enter Feature Values")

input_data = {}
cols = st.columns(3)

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        input_data[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.4f"
        )

# ↓↓↓ THIS BLOCK replaces your old "if st.button(...)" block ↓↓↓
if st.button("🔮 Predict Fault", use_container_width=True):

    if all(v == 0.0 for v in input_data.values()):
        st.warning("⚠️ Please enter actual feature values before predicting — all fields are still at default (0.0).")
    else:
        input_df = pd.DataFrame([input_data], columns=feature_names)
        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        detected_faults = [fault for i, fault in enumerate(fault_names) if prediction[i] == 1]

        if detected_faults:
            st.success("Fault(s) Detected")
            for fault in detected_faults:
                st.write("✅", fault)
        else:
            st.info("No fault detected.")
