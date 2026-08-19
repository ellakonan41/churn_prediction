import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import streamlit as st
import requests
import plotly.graph_objects as go

from src.explanability import get_feature_importance


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000"
)

CLASSIFICATION_THRESHOLD = 0.40


st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📞",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("📞 Customer Churn Intelligence")

st.markdown(
    """
    **Predict • Understand • Retain**

    Identify customers at risk of churn and understand the main
    factors influencing each prediction.
    """
)

st.divider()


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.header("👤 Customer Profile")


# ------------------------------------------------------------
# PERSONAL INFORMATION
# ------------------------------------------------------------

with st.expander(
    "👤 Personal Information",
    expanded=True
):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    with col2:
        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

    with col3:
        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

    with col4:
        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )


# ------------------------------------------------------------
# SERVICES
# ------------------------------------------------------------

with st.expander(
    "📡 Services",
    expanded=True
):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

    with col2:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

    with col3:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    with col4:
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

    with col3:
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

    with col4:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

    col1, col2 = st.columns(2)

    with col1:
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )


# ------------------------------------------------------------
# ACCOUNT INFORMATION
# ------------------------------------------------------------

with st.expander(
    "💳 Account Information",
    expanded=True
):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )

    with col3:
        monthly_charges = st.number_input(
            "Monthly Charges (€)",
            min_value=0.0,
            value=70.0
        )

    with col4:
        total_charges = st.number_input(
            "Total Charges (€)",
            min_value=0.0,
            value=840.0
        )

    col1, col2 = st.columns(2)

    with col1:
        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with col2:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict = st.button(
        "🔮 Predict Customer Churn",
        width="stretch"
    )


# ============================================================
# PREDICTION
# ============================================================

if predict:

    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
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
    }

    try:

        # ----------------------------------------------------
        # API CALL
        # ----------------------------------------------------

        response = requests.post(
            f"{API_URL}/predict",
            json=customer,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        probability = float(
            result["churn_probability"]
        )

        prediction = int(
            result["churn_prediction"]
        )

        # ----------------------------------------------------
        # SHAP EXPLANATION
        # ----------------------------------------------------

        shap_result = get_feature_importance(
            customer
        )

        explanation = shap_result[
            "top_features"
        ]

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()
        st.header("📊 Analysis Result")

        col1, col2 = st.columns([1, 2])

        # ----------------------------------------------------
        # PROBABILITY GAUGE
        # ----------------------------------------------------

        with col1:

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={
                        "suffix": "%"
                    },
                    title={
                        "text": "Churn Probability"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "threshold": {
                            "line": {
                                "width": 4
                            },
                            "value": CLASSIFICATION_THRESHOLD * 100
                        }
                    }
                )
            )

            fig.update_layout(
                height=300,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

            st.caption(
                f"Classification threshold: "
                f"{CLASSIFICATION_THRESHOLD:.0%}"
            )


        # ----------------------------------------------------
        # RISK CARD
        # ----------------------------------------------------

        with col2:

            if prediction == 1:

                st.error(
                    "### 🔴 HIGH CHURN RISK\n\n"
                    "This customer is likely to churn."
                )

                st.markdown(
                    """
                    **🎯 Recommended action**

                    Consider targeting this customer with a
                    retention campaign.
                    """
                )

            else:

                st.success(
                    "### 🟢 LOW CHURN RISK\n\n"
                    "This customer is unlikely to churn."
                )

                st.markdown(
                    """
                    **🎯 Recommended action**

                    No immediate retention action is required.
                    """
                )


        # ====================================================
        # CUSTOMER SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "👤 Customer Summary"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Contract",
                contract
            )

        with col2:
            st.metric(
                "Tenure",
                f"{tenure} months"
            )

        with col3:
            st.metric(
                "Monthly Charges",
                f"€{monthly_charges:.2f}"
            )

        with col4:
            st.metric(
                "Internet",
                internet_service
            )


        # ====================================================
        # WHY THIS PREDICTION?
        # ====================================================

        st.divider()

        st.subheader(
            "🔍 Why this prediction?"
        )

        st.caption(
            "The model considers all customer characteristics. "
            "The five factors below have the strongest individual "
            "impact on this prediction."
        )


        # ----------------------------------------------------
        # OVERALL DRIVERS
        # ----------------------------------------------------

        positive_sum = shap_result[
            "positive_sum"
        ]

        negative_sum = shap_result[
            "negative_sum"
        ]

        driver_col1, driver_col2 = st.columns(2)

        with driver_col1:

            st.metric(
                "🔴 Increasing churn risk",
                f"+{positive_sum:.2f}"
            )

        with driver_col2:

            st.metric(
                "🟢 Reducing churn risk",
                f"{negative_sum:.2f}"
            )


        # ----------------------------------------------------
        # TOP 5 BAR CHART
        # ----------------------------------------------------

        fig2 = go.Figure(
            go.Bar(
                x=explanation["shap_value"],
                y=explanation["feature"],
                orientation="h"
            )
        )

        fig2.update_yaxes(
            autorange="reversed"
        )

        fig2.update_layout(
            title="Top 5 factors influencing this customer",
            xaxis_title="SHAP contribution",
            yaxis_title="Feature",
            height=400,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            fig2,
            width="stretch"
        )


        # ----------------------------------------------------
        # HUMAN-READABLE FACTORS
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        positive_features = explanation[
            explanation["shap_value"] > 0
        ]

        negative_features = explanation[
            explanation["shap_value"] < 0
        ]


        with col1:

            st.markdown(
                "### 🔴 Factors increasing churn risk"
            )

            if positive_features.empty:

                st.write(
                    "No major risk-increasing factors "
                    "among the top 5."
                )

            else:

                for _, row in positive_features.iterrows():

                    st.write(
                        f"🔴 **{row['feature']}** "
                        f"(+{row['shap_value']:.2f})"
                    )


        with col2:

            st.markdown(
                "### 🟢 Factors reducing churn risk"
            )

            if negative_features.empty:

                st.write(
                    "No major risk-reducing factors "
                    "among the top 5."
                )

            else:

                for _, row in negative_features.iterrows():

                    st.write(
                        f"🟢 **{row['feature']}** "
                        f"({row['shap_value']:.2f})"
                    )


        # ====================================================
        # TECHNICAL SHAP DETAILS
        # ====================================================

        with st.expander(
            "🔬 Technical SHAP details"
        ):

            model_probability = shap_result[
                "model_probability"
            ]

            reconstructed_probability = shap_result[
                "reconstructed_probability"
            ]

            difference = abs(
                model_probability
                - reconstructed_probability
            )

            st.write(
                f"**Model probability:** "
                f"{model_probability:.2%}"
            )

            st.write(
                f"**SHAP reconstructed probability:** "
                f"{reconstructed_probability:.2%}"
            )

            st.write(
                f"**Difference:** "
                f"{difference:.6f}"
            )

            st.write(
                f"**Expected value:** "
                f"{shap_result['expected_value']:.4f}"
            )

            st.write(
                f"**Positive SHAP contribution:** "
                f"+{positive_sum:.4f}"
            )

            st.write(
                f"**Negative SHAP contribution:** "
                f"{negative_sum:.4f}"
            )

            st.write(
                f"**Total SHAP contribution:** "
                f"{shap_result['total_shap']:.4f}"
            )

            st.write(
                f"**Classification threshold:** "
                f"{CLASSIFICATION_THRESHOLD:.2%}"
            )

            if difference < 1e-5:

                st.success(
                    "✅ SHAP explanation is consistent "
                    "with the model probability."
                )

            else:

                st.warning(
                    "⚠️ SHAP reconstructed probability "
                    "differs from the model probability."
                )


    except requests.exceptions.RequestException:

        st.error(
            "❌ Unable to connect to the prediction API."
        )

        st.caption(
            f"API endpoint: {API_URL}"
        )

    except Exception as e:

        st.error(
            "❌ An unexpected error occurred."
        )

        st.caption(
            str(e)
        )