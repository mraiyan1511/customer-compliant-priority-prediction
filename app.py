import streamlit as st
import pandas as pd
import joblib

# ==========================================
# CUSTOMER COMPLAINT PRIORITY PREDICTION
# STREAMLIT FRONTEND
# ==========================================

# Page Configuration
st.set_page_config(
    page_title="Customer Complaint Priority Prediction",
    page_icon="📢",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(
        "customer_complaint_priority_model.pkl"
    )

model_data = load_model()

model = model_data["model"]
encoders = model_data["encoders"]
target_encoder = model_data["target_encoder"]
features = model_data["features"]

# ==========================================
# TITLE
# ==========================================

st.title("📢 Customer Complaint Priority Prediction")

st.write(
    "### Random Forest Based Prediction System"
)

st.markdown(
    "Enter the customer complaint details below "
    "to predict its priority."
)

st.divider()

# ==========================================
# INPUT FORM
# ==========================================

with st.form("complaint_form"):

    st.subheader("📝 Complaint Details")

    col1, col2 = st.columns(2)

    with col1:

        product = st.text_input(
            "Product",
            placeholder="Example: Credit card"
        )

        sub_product = st.text_input(
            "Sub-product",
            placeholder="Example: General-purpose credit card"
        )

        issue = st.text_input(
            "Issue",
            placeholder="Example: Billing dispute"
        )

        sub_issue = st.text_input(
            "Sub-issue",
            placeholder="Example: Wrong amount charged"
        )

        company = st.text_input(
            "Company",
            placeholder="Example: Bank"
        )

    with col2:

        state = st.text_input(
            "State",
            placeholder="Example: CA"
        )

        submitted_via = st.selectbox(
            "Submitted via",
            [
                "Web",
                "Phone",
                "Email",
                "Referral",
                "Postal mail",
                "Other"
            ]
        )

        company_response = st.selectbox(
            "Company response to consumer",
            [
                "Closed with explanation",
                "Closed with monetary relief",
                "Closed",
                "In progress",
                "Other"
            ]
        )

        timely_response = st.selectbox(
            "Timely response?",
            ["Yes", "No"]
        )

        consumer_disputed = st.selectbox(
            "Consumer disputed?",
            ["Yes", "No"]
        )

    submit = st.form_submit_button(
        "🔮 Predict Priority",
        use_container_width=True
    )

# ==========================================
# PREDICTION
# ==========================================

if submit:

    input_data = pd.DataFrame({
        "Product": [product],
        "Sub-product": [sub_product],
        "Issue": [issue],
        "Sub-issue": [sub_issue],
        "Company": [company],
        "State": [state],
        "Submitted via": [submitted_via],
        "Company response to consumer": [company_response],
        "Timely response?": [timely_response],
        "Consumer disputed?": [consumer_disputed]
    })

    # Fill empty values
    for column in input_data.columns:

        input_data[column] = (
            input_data[column]
            .fillna("Unknown")
            .astype(str)
        )

    # Encode input
    for column in input_data.columns:

        encoder = encoders[column]

        known_values = set(
            encoder.classes_
        )

        input_data[column] = input_data[
            column
        ].apply(
            lambda value:
            value
            if value in known_values
            else encoder.classes_[0]
        )

        input_data[column] = encoder.transform(
            input_data[column]
        )

    # Prediction
    prediction = model.predict(
        input_data[features]
    )

    priority = target_encoder.inverse_transform(
        prediction
    )[0]

    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.divider()

    st.subheader("🎯 Prediction Result")

    if priority == "High":

        st.error(
            f"🔴 Priority: {priority}"
        )

    elif priority == "Medium":

        st.warning(
            f"🟡 Priority: {priority}"
        )

    else:

        st.success(
            f"🟢 Priority: {priority}"
        )

    st.info(
        "The complaint priority was predicted "
        "using the trained Random Forest model."
    )