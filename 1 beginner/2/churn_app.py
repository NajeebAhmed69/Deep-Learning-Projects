import numpy as np
import pandas as pd
import joblib
import streamlit as st
from tensorflow import keras

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details to estimate the probability that the customer will churn."
)

# Load saved model, preprocessor, and threshold
@st.cache_resource
def load_assets():
    model = keras.models.load_model("churn_ann_model.keras")
    preprocessor = joblib.load("churn_preprocessor.joblib")
    threshold = joblib.load("churn_best_threshold.joblib")

    return model, preprocessor, float(threshold)


model, preprocessor, saved_threshold = load_assets()

# Sidebar inputs
st.sidebar.header("Customer details")

gender = st.sidebar.selectbox(
    "Gender",
    options=["Female", "Male"]
)

senior_citizen = st.sidebar.selectbox(
    "Senior citizen",
    options=["No", "Yes"]
)

partner = st.sidebar.selectbox(
    "Has partner",
    options=["No", "Yes"]
)

dependents = st.sidebar.selectbox(
    "Has dependents",
    options=["No", "Yes"]
)

tenure = st.sidebar.slider(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)

phone_service = st.sidebar.selectbox(
    "Phone service",
    options=["No", "Yes"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple lines",
    options=["No phone service", "No", "Yes"]
)

internet_service = st.sidebar.selectbox(
    "Internet service",
    options=["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online security",
    options=["No internet service", "No", "Yes"]
)

online_backup = st.sidebar.selectbox(
    "Online backup",
    options=["No internet service", "No", "Yes"]
)

device_protection = st.sidebar.selectbox(
    "Device protection",
    options=["No internet service", "No", "Yes"]
)

tech_support = st.sidebar.selectbox(
    "Tech support",
    options=["No internet service", "No", "Yes"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    options=["No internet service", "No", "Yes"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming movies",
    options=["No internet service", "No", "Yes"]
)

contract = st.sidebar.selectbox(
    "Contract",
    options=[
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless billing",
    options=["No", "Yes"]
)

payment_method = st.sidebar.selectbox(
    "Payment method",
    options=[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.sidebar.number_input(
    "Monthly charges ($)",
    min_value=0.0,
    max_value=200.0,
    value=70.0,
    step=0.05
)

total_charges = st.sidebar.number_input(
    "Total charges ($)",
    min_value=0.0,
    max_value=10000.0,
    value=1500.0,
    step=0.05
)

decision_threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.0,
    max_value=1.0,
    value=saved_threshold,
    step=0.01
)

# Convert categorical values into the expected training format
senior_citizen_value = 1 if senior_citizen == "Yes" else 0

# Create a one-row customer DataFrame
customer_input = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior_citizen_value,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])

# Prediction function
def predict_churn(input_data, threshold):
    processed_input = preprocessor.transform(input_data)

    probability = model.predict(
        processed_input,
        verbose=0
    ).ravel()[0]

    probability = float(np.clip(probability, 0.0, 1.0))

    prediction = int(probability >= threshold)

    return probability, prediction


# Single customer prediction
st.header("Single customer prediction")

if st.button("Predict churn risk", type="primary"):
    try:
        churn_probability, churn_prediction = predict_churn(
            customer_input,
            decision_threshold
        )

        prediction_text = (
            "Churn" if churn_prediction == 1 else "No churn"
        )

        st.subheader("Prediction result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Predicted class",
                value=prediction_text
            )

        with col2:
            st.metric(
                label="Churn probability",
                value=f"{churn_probability:.2%}"
            )

        st.progress(
            value=churn_probability,
            text=f"Churn probability: {churn_probability:.2%}"
        )

        st.info(
            f"Decision threshold: **{decision_threshold:.2f}**"
        )

        if churn_prediction == 1:
            st.warning(
                "This customer is predicted as likely to churn. "
                "Consider a retention offer or proactive support."
            )
        else:
            st.success(
                "This customer is predicted as unlikely to churn."
            )

        with st.expander("See entered customer details"):
            st.dataframe(customer_input, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# Batch prediction
st.header("Batch prediction")
st.write(
    "Upload a CSV file containing customer rows. "
    "The file may include `customerID` and `Churn`; both will be removed before prediction."
)

uploaded_file = st.file_uploader(
    "Upload customer CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)

        if "customerID" in batch_data.columns:
            batch_data = batch_data.drop(columns=["customerID"])

        if "Churn" in batch_data.columns:
            batch_data = batch_data.drop(columns=["Churn"])

        batch_processed = preprocessor.transform(batch_data)

        batch_probabilities = model.predict(
            batch_processed,
            verbose=0
        ).ravel()

        batch_probabilities = np.clip(
            batch_probabilities,
            0.0,
            1.0
        )

        batch_predictions = (
            batch_probabilities >= decision_threshold
        ).astype(int)

        result_df = batch_data.copy()
        result_df["churn_probability"] = batch_probabilities
        result_df["predicted_churn"] = batch_predictions
        result_df["prediction"] = np.where(
            batch_predictions == 1,
            "Churn",
            "No churn"
        )

        st.subheader("Batch prediction results")

        churn_count = int((batch_predictions == 1).sum())
        no_churn_count = int((batch_predictions == 0).sum())

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Customers predicted to churn",
                value=churn_count
            )

        with col2:
            st.metric(
                label="Customers predicted not to churn",
                value=no_churn_count
            )

        st.dataframe(result_df, use_container_width=True)

        csv_data = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download predictions as CSV",
            data=csv_data,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"Batch prediction failed: {error}")

# Footer
st.divider()
st.caption(
    "This app estimates churn risk from the trained model. "
    "It does not guarantee that a customer will churn."
)