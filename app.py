
import streamlit as st
import pandas as pd
import joblib
import json
import os
from pathlib import Path

# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "deploy" / "cardio_model.joblib"
META_PATH = BASE_DIR / "deploy" / "meta.json"


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run the corrected notebook deployment cell first."
        )

    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {META_PATH}"
        )

    with open(META_PATH, "r") as file:
        metadata = json.load(file)

    model = joblib.load(MODEL_PATH)

    return model, metadata


try:
    model, metadata = load_model()

except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()


# ==========================================
# 3. APPLICATION HEADER
# ==========================================

st.title("❤️ Cardiovascular Disease Prediction")

st.markdown(
    """
    ### Machine Learning Based Prediction System

    This application uses a **Gradient Boosting Classifier**
    to predict the presence or absence of cardiovascular
    disease based on patient health information.
    """
)

st.divider()

st.info(
    "Enter the patient's health information below "
    "to obtain a machine learning prediction."
)


# ==========================================
# 4. PATIENT INPUT FORM
# ==========================================

st.header("👤 Patient Information")

with st.form("cardio_prediction_form"):

    st.subheader("Basic Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=50,
            step=1
        )

        gender = st.selectbox(
            "Gender (dataset code)",
            options=["male", "female"]
        )

    with col2:

        height = st.number_input(
            "Height (cm)",
            min_value=120.0,
            max_value=220.0,
            value=170.0,
            step=1.0
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=200.0,
            value=70.0,
            step=0.5
        )

    st.divider()

    st.subheader("Blood Pressure Information")

    col3, col4 = st.columns(2)

    with col3:

        ap_hi = st.number_input(
            "Systolic Blood Pressure (ap_hi)",
            min_value=70,
            max_value=250,
            value=120
        )

    with col4:

        ap_lo = st.number_input(
            "Diastolic Blood Pressure (ap_lo)",
            min_value=40,
            max_value=200,
            value=80
        )

    st.caption(
        "Blood pressure limits are input validation limits "
        "used in this project, not medical normal ranges."
    )

    st.divider()

    st.subheader("Medical Information")

    col5, col6 = st.columns(2)

    with col5:

        cholesterol = st.selectbox(
            "Cholesterol Level",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[x]
        )

    with col6:

        gluc = st.selectbox(
            "Glucose Level",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[x]
        )

    st.divider()

    st.subheader("Lifestyle Information")

    col7, col8 = st.columns(2)

    with col7:

        smoke = st.selectbox(
            "Smoking",
            options=[0, 1],
            format_func=lambda x: {
                0: "No",
                1: "Yes"
            }[x]
        )

        alco = st.selectbox(
            "Alcohol Intake",
            options=[0, 1],
            format_func=lambda x: {
                0: "No",
                1: "Yes"
            }[x]
        )

    with col8:

        active = st.selectbox(
            "Physical Activity",
            options=[0, 1],
            format_func=lambda x: {
                0: "No",
                1: "Yes"
            }[x]
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Cardiovascular Disease",
        use_container_width=True
    )


# ==========================================
# 5. PREDICTION PROCESS
# ==========================================

if submitted:

    # Validate blood pressure
    if ap_hi <= ap_lo:

        st.error(
            "Invalid blood pressure! "
            "Systolic pressure must be greater "
            "than diastolic pressure."
        )

    else:

        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)

        # Age is already entered in years.
        # Notebook uses age_years rounded to one decimal.
        age_years = round(float(age), 1)

        # Prepare patient input
        input_data = {

            "gender": gender,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke,
            "alco": alco,
            "active": active,
            "age_years": age_years,
            "bmi": bmi
        }

        # Maintain the exact feature order used in training
        feature_order = metadata["feature_order"]

        # Check that all required features are available
        missing_features = [
            feature for feature in feature_order
            if feature not in input_data
        ]

        if missing_features:

            st.error(
                f"Missing model features: {missing_features}"
            )

        else:

            input_df = pd.DataFrame(
                [input_data],
                columns=feature_order
            )

            try:

                # Predict using the complete pipeline
                prediction = model.predict(input_df)[0]

                # Calculate probability for class 1
                probabilities = model.predict_proba(input_df)[0]

                positive_index = list(
                    model.classes_
                ).index(1)

                probability = (
                    probabilities[positive_index] * 100
                )

                # ==================================
                # 6. DISPLAY RESULTS
                # ==================================

                st.divider()

                st.header("📊 Prediction Result")

                col9, col10 = st.columns(2)

                with col9:

                    st.metric(
                        "Calculated BMI",
                        f"{bmi:.2f}"
                    )

                with col10:

                    st.metric(
                        "Class 1 probability",
                        f"{probability:.2f}%"
                    )

                st.subheader("Model Prediction")

                if prediction == 1:

                    st.error(
                        "⚠️ Cardiovascular Disease Predicted"
                    )

                    st.write(
                        "The model predicts class 1 "
                        "(cardiovascular disease present)."
                    )

                else:

                    st.success(
                        "✅ No Cardiovascular Disease Predicted"
                    )

                    st.write(
                        "The model predicts class 0 "
                        "(cardiovascular disease absent)."
                    )

                # Probability visualization
                st.subheader("Prediction Probability")

                st.progress(
                    int(round(probability))
                )

                st.write(
                    f"Estimated probability for class 1: "
                    f"**{probability:.2f}%**"
                )

                st.caption(
                    "This is the model's estimated probability "
                    "for the positive dataset class. It is not "
                    "a clinically validated estimate of "
                    "individual cardiovascular risk."
                )

                # Patient summary
                with st.expander("View Patient Information"):

                    st.dataframe(
                        input_df,
                        use_container_width=True
                    )

                st.divider()

                st.warning(
                    "Medical Disclaimer: This application is "
                    "developed for academic and educational "
                    "purposes. Its predictions are not a medical "
                    "diagnosis and must not be used to make "
                    "treatment decisions. Consult a qualified "
                    "healthcare professional for medical advice."
                )

            except Exception as e:

                st.error(
                    "An error occurred during prediction."
                )

                st.exception(e)


# ==========================================
# 7. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center">

    **Cardiovascular Disease Prediction System**

    Machine Learning Mini Project | MCA

    Developed using Python, Scikit-learn and Streamlit

    </div>
    """,
    unsafe_allow_html=True
)
