import streamlit as st
import pandas as pd
import joblib

# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# --------------------------------
# 2. Load Trained Model
# --------------------------------

@st.cache_resource
def load_model():
    return joblib.load("predictive_maintenance_model.pkl")

model = load_model()

# --------------------------------
# 3. Application Title
# --------------------------------

st.title("⚙️ AI Predictive Maintenance System")

st.write(
    "Predict potential machine failure using sensor readings "
    "and a trained Machine Learning model."
)

st.info("Prediction threshold: 0.30")

# --------------------------------
# 4. User Input
# --------------------------------

st.subheader("Enter Machine Sensor Readings")

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temp = st.number_input(
        "Air Temperature (K)",
        min_value=250.0,
        max_value=350.0,
        value=300.0
    )

    process_temp = st.number_input(
        "Process Temperature (K)",
        min_value=250.0,
        max_value=400.0,
        value=310.0
    )

with col2:
    rotational_speed = st.number_input(
        "Rotational Speed (rpm)",
        min_value=0,
        max_value=10000,
        value=1500
    )

    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=200.0,
        value=45.0
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        min_value=0,
        max_value=500,
        value=120
    )

# --------------------------------
# 5. Prediction
# --------------------------------

if st.button("Predict Machine Failure", type="primary"):

    input_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    # Predict failure probability estimate
    probability = model.predict_proba(input_data)[0][1]

    # Selected threshold
    threshold = 0.30

    st.subheader("Prediction Results")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            "Estimated Failure Probability",
            f"{probability * 100:.2f}%"
        )

    with metric2:
        st.metric(
            "Prediction Threshold",
            f"{threshold * 100:.0f}%"
        )

    if probability >= threshold:
        st.error(
            "⚠️ Potential Machine Failure Detected!"
        )

        st.warning(
            "Consider inspecting the machine. "
            "This prediction is not a confirmed diagnosis."
        )

    else:
        st.success(
            "✅ Model predicts No Failure"
        )

        st.info(
            "Continue normal monitoring and follow "
            "your maintenance schedule."
        )

    # Show the entered readings
    with st.expander("View Sensor Readings"):
        st.dataframe(input_data, use_container_width=True)

# --------------------------------
# 6. Disclaimer
# --------------------------------

st.divider()

st.caption(
    "Educational ML project. Predictions are estimates based "
    "on the training dataset and are not guaranteed to reflect "
    "real-world machine conditions."
)